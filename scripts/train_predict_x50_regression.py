"""前処理済みの x50 データ（long 型）を、時系列ではなく通常の回帰として1つのモデルで学習・予測する。

`scripts/train_predict_x50_global.py`（グローバルモデルの再帰予測）の比較用。入力の読み込み・確認、
図の部品は `scripts/train_predict_x50.py` と同じものを使う。

- 時系列として扱わない: 目的変数のラグ・移動平均・前日比は使わない（テスト期間の値が分からず、
  再帰予測もしないため作れない）。特徴量は「系列ID（`Y_cat`）+ 日付の特徴量 + 説明変数」だけ。
  各行は、その行の特徴量だけから予測する。
- 1つのモデルで全種類（`Y_cat` = Y1〜Y10）を学習する（グローバルモデルと同じ）。系列IDは
  `feature_engineering.categorical.PandasOrdinalEncoder` で整数にする
  （CVの fold ごとに学習側で fit）。
- 目的変数は元の尺度（`Y_raw`）のまま学習・予測する。目的変数が欠損の行は学習できないため除く。
- CV は GroupKFold（`cv.method="group"`）。グループは年月（`year_month`）で、同じ月の行
  （全種類・全日）は必ず同じ fold に入る。隣り合う日は値が似ている（自己相関）ため、日付単位で
  分けるより検証が甘くなりにくい。月は fold にランダムに割り当てる。
- 正解ファイルは、テスト予測が終わった後の評価にだけ使う。

注意: GroupKFold では、検証の月より後の月も学習に使う。テスト（学習データより後の2025年）とは
条件が違うため、CV の誤差はテストの誤差より小さく出やすい。

出力（`outputs/experiments/x50_regression_{モデル名}/{実行日時}/`）:
    - `run_experiment` の出力一式（CVスコア・OOF/テスト予測・誤差評価の図・SHAP・設定）
    - `summary.csv`（種類ごとの MAE）、`test_predictions.csv`（全種類のテスト予測。long 型）、
      `shap_importance_by_target.csv`
    - `figures/`: `summary_mae.png`・`training_overview.png`・`test_overview.png`・
      `shap_overview.png`
    - `run.log`。MLflow（`mlruns/`）にも指標・図・ログを記録する。

使い方:
    uv run python scripts/train_predict_x50_regression.py --model lightgbm
    uv run python scripts/train_predict_x50_regression.py --model xgboost --n-splits 4
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import train_predict_x50 as base  # noqa: E402

from modeling.config import ExperimentConfig  # noqa: E402
from modeling.dataset import Dataset  # noqa: E402
from modeling.experiment import ExperimentResult, prepare_dataset, run_experiment  # noqa: E402
from modeling.metrics import get_metric  # noqa: E402
from modeling.tracking import Tracker  # noqa: E402
from util.paths import ensure_parent_dir, outputs_dir  # noqa: E402
from util.plotting import add_caption, ensure_japanese_font  # noqa: E402

logger = logging.getLogger("train_predict_x50_regression")

GROUP_COL = "year_month"  # GroupKFold のグループ（年月。例: 2022-01）
DEFAULT_N_SPLITS = 5  # GroupKFold の分割数


@dataclass
class RegressionRun:
    """通常の回帰モデルの学習・予測の結果。

    Attributes:
        categories: 目的変数の種類（系列）の並び。
        dataset: 学習用データ一式。
        frame: 学習に使った行の表（`dataset.X` と同じ行順。列 `date`, `Y_cat`, `year_month`, `Y`）。
        result: `run_experiment` の結果。
        predictions: テスト予測（long 型。列 `date`, `Y_cat`, `pred`。正解があれば `actual` も）。
        oof: 学習期間の OOF 予測（long 型。列 `date`, `Y_cat`, `actual`, `pred`, `fold`）。
    """

    categories: list[str]
    dataset: Dataset
    frame: pd.DataFrame
    result: ExperimentResult
    predictions: pd.DataFrame
    oof: pd.DataFrame


# --- モデルに渡す表 ---------------------------------------------------------------------


def model_frames(inputs: base.Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`prepare_dataset` に渡す学習・テストの表を作る（long 型のまま、全種類の行を使う）。

    学習の表は「日付・系列ID（`Y_cat`）・年月（グループ）・日付の特徴量・説明変数・目的変数（元の尺度）」。
    CSVにある目的変数の特徴量（`Y_lag_*`）は使わない（テスト期間には作れないため）。
    目的変数が欠損の行は学習できないため除く。

    Returns:
        (学習の表, テストの表（目的変数なし）)。
    """
    columns = [base.DATE_COL, base.CAT_COL, *inputs.exog_cols]
    dates = pd.to_datetime(inputs.train[base.DATE_COL])
    train = inputs.train[columns].assign(
        **{
            GROUP_COL: dates.dt.strftime("%Y-%m"),
            base.TARGET_COL: inputs.train[f"{base.TARGET_COL}{base.RAW_SUFFIX}"],
        }
    )
    train = train[train[base.TARGET_COL].notna()]
    test = inputs.test[columns]
    assert base.TARGET_COL not in test.columns
    return train.reset_index(drop=True), test.reset_index(drop=True)


# --- 実験設定 ---------------------------------------------------------------------------


def make_config(
    model: str,
    n_series: int,
    n_splits: int = DEFAULT_N_SPLITS,
    n_trials: int = 0,
    test_prediction: str = "refit_full",
) -> ExperimentConfig:
    """通常の回帰モデルの実験設定を作る（`ExperimentConfig` で検証する）。

    Args:
        model: モデル名（`lightgbm` / `xgboost`）。
        n_series: 種類（系列）の数（記録用）。
        n_splits: GroupKFold の分割数。
        n_trials: Optunaの試行回数（0ならチューニングしない）。
        test_prediction: テスト予測の作り方（`refit_full` / `fold_mean`）。

    Raises:
        ValueError: 未対応のモデル名の場合。
    """
    if model not in base.MODELS:
        raise ValueError(f"モデルは {base.MODELS} から選んでください: {model}")
    return ExperimentConfig.model_validate(
        {
            "name": f"x50_regression_{model}",
            # 時系列として扱わない通常の回帰（目的変数のラグ・再帰予測を使わない）
            "task": "regression",
            "data": {
                "train_path": str(base.TRAIN_PATH),
                "test_path": str(base.TEST_PATH),
                "target": base.TARGET_COL,
                # GroupKFold のグループ。日付・グループの列はモデルの特徴量から外す
                "group_col": GROUP_COL,
                "drop_cols": [base.DATE_COL, GROUP_COL],
            },
            # 系列ID（Y1〜Y10 の文字列）を整数にして、モデルが種類を区別できるようにする。
            # CVの各 fold の学習側だけで fit される
            "features": [
                {
                    "class": "feature_engineering.categorical.PandasOrdinalEncoder",
                    "params": {"variables": [base.CAT_COL]},
                }
            ],
            # 年月をグループにした GroupKFold（同じ月の行は学習と検証に分かれない）
            "cv": {"method": "group", "n_splits": n_splits},
            "model": {"name": model, "params": {"n_estimators": 1000}, "early_stopping_rounds": 50},
            "metrics": ["mae"],
            "test_prediction": test_prediction,
            "tuning": {"enabled": n_trials > 0, "n_trials": max(n_trials, 1)},
            "explain": {"enabled": True},
            "evaluation": {"enabled": True, "training_history": True, "max_series": n_series},
            "tracking": {"experiment_name": base.MLFLOW_EXPERIMENT},
        }
    )


# --- 学習と確認 -------------------------------------------------------------------------


def prepare(config: ExperimentConfig, inputs: base.Inputs) -> tuple[Dataset, pd.DataFrame]:
    """全種類の学習用データ一式を作り、リークのおそれがないことを確かめる。

    確かめること:
        - モデルの特徴量に、目的変数（`Y`・`Y_raw`）・目的変数の特徴量（`Y_lag_*`）・
          日付・グループの列が無い。
        - CVの各 fold で、同じ年月が学習と検証の両方に入っていない。
          各年月はちょうど1つの fold の検証に入る。
        - テストデータに目的変数が無い。

    Returns:
        (学習用データ一式, 学習に使った行の表（`dataset.X` と同じ行順）)。

    Raises:
        base.LeakageError: 上のいずれかを満たさない場合。
    """
    train, test = model_frames(inputs)
    n_dropped = len(inputs.train) - len(train)
    dataset = prepare_dataset(config, train, test)

    target = base.TARGET_COL
    leaked = [
        c
        for c in dataset.X.columns
        if c in (target, f"{target}{base.RAW_SUFFIX}", base.DATE_COL, GROUP_COL)
        or c.startswith(f"{target}_lag_")
    ]
    if leaked:
        raise base.LeakageError(f"モデルの特徴量に目的変数・日付・グループの列があります: {leaked}")

    groups = train[GROUP_COL].to_numpy()
    validated: list[str] = []
    for i, (train_idx, valid_idx) in enumerate(dataset.folds):
        shared = set(groups[train_idx]) & set(groups[valid_idx])
        if shared:
            raise base.LeakageError(f"fold {i}: 同じ年月が学習と検証の両方にあります: {shared}")
        months = sorted(set(groups[valid_idx]))
        validated += months
        logger.info(
            "fold %d: 検証 %d か月（%s）・%d行 / 学習 %d行",
            i,
            len(months),
            ", ".join(months),
            len(valid_idx),
            len(train_idx),
        )
    if sorted(validated) != sorted(set(groups)):
        raise base.LeakageError("検証に入らない年月、または複数の fold の検証に入る年月があります")
    if dataset.X_test is None or target in dataset.X_test.columns:
        raise base.LeakageError(
            "テストデータに目的変数があります（またはテストデータがありません）"
        )
    logger.info(
        "学習データ（long 型）: %d行（目的変数が欠損の %d行を除く）,"
        " 特徴量 %d列（系列ID・日付・説明変数）",
        len(dataset.X),
        n_dropped,
        dataset.X.shape[1],
    )
    return dataset, train


def train_and_predict(
    config: ExperimentConfig,
    dataset: Dataset,
    tracker: Tracker,
    output_root: Path | None = None,
    extra_tags: dict[str, str] | None = None,
) -> ExperimentResult:
    """GroupKFold の CV 学習・テスト予測・誤差評価・SHAP を `run_experiment` で行う。"""
    logger.info(
        "学習開始: 通常の回帰モデル=%s, CV=GroupKFold（%d分割、グループ=%s）,"
        " チューニング=%s, テスト予測=%s",
        config.model.name,
        config.cv.n_splits,
        GROUP_COL,
        f"{config.tuning.n_trials}試行" if config.tuning.enabled else "なし",
        config.test_prediction,
    )
    start = time.perf_counter()
    result = run_experiment(
        config,
        dataset,
        tracker=tracker,
        output_root=output_root or outputs_dir() / "experiments",
        extra_tags={"model_type": "regression", **(extra_tags or {})},
    )
    cv = result.cv_result
    logger.info(
        "学習終了（%.1f秒）: CV MAE fold別 %s / OOF %.4f。出力 %s",
        time.perf_counter() - start,
        [round(s, 4) for s in cv.fold_scores["mae"]],
        cv.oof_scores["mae"],
        base.shown(result.output_dir),
    )
    if result.tuning_result is not None:
        logger.info("チューニングの最良パラメータ: %s", result.tuning_result.best_params)
    return result


def collect_predictions(
    categories: list[str],
    dataset: Dataset,
    frame: pd.DataFrame,
    test: pd.DataFrame,
    result: ExperimentResult,
    answer: pd.DataFrame | None,
) -> RegressionRun:
    """テスト予測と OOF 予測を、日付・種類つきの long 型の表にする（正解があれば付ける）。

    Args:
        categories: 種類の並び。
        dataset: 学習用データ一式。
        frame: `prepare` が返した学習の表（`dataset.X` と同じ行順）。
        test: テストデータ（`dataset.X_test` と同じ行順。`inputs.test`）。
        result: `run_experiment` の結果。
        answer: テストの正解（Noneなら付けない）。正解の値が欠損の行は評価から除く。

    Raises:
        ValueError: 正解ファイルに、テスト期間の一部の日付・種類の行が無い場合。
    """
    cv = result.cv_result
    assert cv.test_pred is not None
    keys = [base.DATE_COL, base.CAT_COL]
    predictions = test[keys].reset_index(drop=True).assign(**{base.PRED_COL: cv.test_pred})
    if answer is not None:
        # 正解（long 型）はテスト予測が終わった後にだけ使う
        actual = answer[[*keys, base.TARGET_COL]].rename(columns={base.TARGET_COL: base.ACTUAL_COL})
        predictions = predictions.merge(actual, on=keys, how="left", validate="1:1", indicator=True)
        if (predictions["_merge"] != "both").any():
            raise ValueError("正解ファイルに、テスト期間の一部の日付・種類がありません")
        predictions = predictions.drop(columns="_merge")
    oof = (
        frame[keys]
        .reset_index(drop=True)
        .assign(**{base.ACTUAL_COL: dataset.y, base.PRED_COL: cv.oof_pred, "fold": cv.fold_ids})
    )
    return RegressionRun(categories, dataset, frame, result, predictions, oof)


# --- 要約と図 ---------------------------------------------------------------------------


def summary_frame(run: RegressionRun) -> pd.DataFrame:
    """種類ごとの MAE をまとめた表（`train_predict_x50.summary_frame` と同じ列）。

    正解が欠損の行は MAE・相対MAE の計算から除く。
    """
    mae = get_metric("mae")
    actual_col, pred_col = base.ACTUAL_COL, base.PRED_COL
    rows = []
    for category in run.categories:
        oof = run.oof[(run.oof[base.CAT_COL] == category) & (run.oof["fold"] >= 0)]
        cv_mae = mae(oof[actual_col].to_numpy(), oof[pred_col].to_numpy())
        row: dict[str, object] = {
            base.CAT_COL: category,
            "cv_oof_mae": cv_mae,
            "cv_relative_mae": cv_mae / float(oof[actual_col].mean()),
        }
        if actual_col in run.predictions.columns:
            test = run.predictions[run.predictions[base.CAT_COL] == category]
            test_mae = mae(test[actual_col].to_numpy(), test[pred_col].to_numpy())
            row["test_mae"] = test_mae
            row["test_relative_mae"] = test_mae / float(test[actual_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_training_overview(run: RegressionRun, title: str, ncols: int = 2) -> plt.Figure:
    """学習結果: 系列ごとに、学習期間の実測値と OOF 予測（GroupKFold の検証の予測）を並べて描く。"""
    ensure_japanese_font()
    nrows = int(np.ceil(len(run.categories) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 2.6 * nrows), constrained_layout=True, sharex=True, squeeze=False
    )
    mae = get_metric("mae")
    date_col, actual_col, pred_col = base.DATE_COL, base.ACTUAL_COL, base.PRED_COL
    for ax, category in zip(axes.flat, run.categories, strict=False):
        part = run.oof[run.oof[base.CAT_COL] == category]
        ax.plot(part[date_col], part[actual_col], color="black", linewidth=0.7, label="実測")
        ax.plot(part[date_col], part[pred_col], color="C3", linewidth=0.9, label="OOF予測")
        score = mae(part[actual_col].to_numpy(), part[pred_col].to_numpy())
        ax.set_title(f"{category}（CV MAE={score:.2f}）")
    for ax in list(axes.flat)[len(run.categories) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(f"{title}: 学習期間の実測値と、OOF 予測（系列ごと）")
    add_caption(
        fig,
        f"1つのモデルで全種類を学習。各月の予測は、その月を含まない fold のモデルによる"
        f"（GroupKFold、グループ={GROUP_COL}）。目的変数が欠損の日は除く。",
    )
    return fig


def plot_test_overview(run: RegressionRun, title: str, ncols: int = 2) -> plt.Figure:
    """テスト結果: 系列ごとに、テスト期間の予測（正解があれば実測も）を並べて描く。"""
    ensure_japanese_font()
    nrows = int(np.ceil(len(run.categories) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 2.6 * nrows), constrained_layout=True, sharex=True, squeeze=False
    )
    date_col, actual_col, pred_col = base.DATE_COL, base.ACTUAL_COL, base.PRED_COL
    has_answer = actual_col in run.predictions.columns
    mae = get_metric("mae")
    for ax, category in zip(axes.flat, run.categories, strict=False):
        part = run.predictions[run.predictions[base.CAT_COL] == category]
        label = category
        if has_answer:
            ax.plot(part[date_col], part[actual_col], color="black", linewidth=0.8, label="実測")
            label += f"（MAE={mae(part[actual_col].to_numpy(), part[pred_col].to_numpy()):.2f}）"
        ax.plot(part[date_col], part[pred_col], color="C3", linewidth=1.2, label="予測")
        ax.set_title(label)
    for ax in list(axes.flat)[len(run.categories) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(f"{title}: テスト期間の予測（系列ごと）")
    return fig


def shap_importance_table(run: RegressionRun) -> pd.DataFrame:
    """系列ごとの SHAP 重要度（平均|SHAP値|の割合）を、特徴量 × 種類の表にする。

    OOF SHAP 値を行の系列（種類）で分けて平均する。系列ID（`Y_cat`）は `系列ID` に名前を変える。
    """
    values = pd.read_parquet(run.result.output_dir / "shap" / "shap_values.parquet")
    series = run.frame[base.CAT_COL].to_numpy()[values.pop("row").to_numpy()]
    rename = {c: "系列ID" if c == base.CAT_COL else c for c in values.columns}
    importance = values.abs().groupby(series).mean().T.rename(index=rename)
    importance = importance[[c for c in run.categories if c in importance.columns]]
    return importance / importance.sum()


def save_outputs(
    run: RegressionRun, model: str, tracker_enabled: bool, log_path: Path | None = None
) -> pd.DataFrame:
    """要約の表・予測・図・ログを実験の出力ディレクトリに保存し、MLflow の同じ run にも記録する。"""
    out = run.result.output_dir
    title = f"x50_regression_{model}"
    summary = summary_frame(run)
    paths = [ensure_parent_dir(out / "summary.csv")]
    summary.to_csv(paths[0], index=False)
    paths.append(out / "test_predictions.csv")
    run.predictions.to_csv(paths[-1], index=False, date_format="%Y-%m-%d")
    paths.append(out / "shap_importance_by_target.csv")
    shap_table = shap_importance_table(run)
    shap_table.to_csv(paths[-1], index_label="feature")
    figures = out / "figures"
    paths.append(
        base.save_figure(base.plot_summary_mae(summary, title), figures / "summary_mae.png")
    )
    paths.append(
        base.save_figure(plot_training_overview(run, title), figures / "training_overview.png")
    )
    paths.append(base.save_figure(plot_test_overview(run, title), figures / "test_overview.png"))
    paths.append(
        base.save_figure(base.plot_shap_overview(shap_table, title), figures / "shap_overview.png")
    )

    message = f"要約: CV 相対MAE の平均 {float(summary['cv_relative_mae'].mean()):.4f}"
    if "test_relative_mae" in summary.columns:
        message += f" / テスト 相対MAE の平均 {float(summary['test_relative_mae'].mean()):.4f}"
    logger.info("%s（%d種類）。表: %s", message, len(run.categories), base.shown(paths[0]))
    if log_path is not None:
        for handler in logging.getLogger().handlers:
            handler.flush()
        paths.append(Path(shutil.copy(log_path, out / "run.log")))

    # run_experiment のrun（終了済み）に、種類ごとの指標と追加の出力を記録する
    if tracker_enabled and run.result.run_id is not None:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        metrics = {"mean_cv_relative_mae": float(summary["cv_relative_mae"].mean())}
        for row in summary.to_dict("records"):
            metrics[f"cv_oof_mae_{row[base.CAT_COL]}"] = float(row["cv_oof_mae"])
            if "test_mae" in row:
                metrics[f"test_mae_{row[base.CAT_COL]}"] = float(row["test_mae"])
        if "test_relative_mae" in summary.columns:
            metrics["mean_test_relative_mae"] = float(summary["test_relative_mae"].mean())
        for key, value in metrics.items():
            client.log_metric(run.result.run_id, key, value)
        for path in paths:
            client.log_artifact(run.result.run_id, str(path), artifact_path="summary")
        logger.info("MLflow に記録: run_id=%s", run.result.run_id)
    return summary


# --- メイン処理 -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """通常の回帰モデルの学習・テスト予測・評価・図・ログの保存を行う。"""
    import matplotlib

    # 図はファイルに保存するだけなので、画面を使わないバックエンドにする
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=base.MODELS, default="lightgbm", help="モデル")
    parser.add_argument(
        "--categories", nargs="+", default=None, help="目的変数の種類（既定: 全種類）"
    )
    parser.add_argument(
        "--n-splits", type=int, default=DEFAULT_N_SPLITS, help="GroupKFold の分割数"
    )
    parser.add_argument(
        "--n-trials", type=int, default=0, help="Optunaの試行回数（0ならチューニングしない）"
    )
    parser.add_argument(
        "--test-prediction", choices=["refit_full", "fold_mean"], default="refit_full"
    )
    parser.add_argument("--train", type=Path, default=base.TRAIN_PATH, help="学習データのCSV")
    parser.add_argument("--test", type=Path, default=base.TEST_PATH, help="テストデータのCSV")
    parser.add_argument(
        "--answer", type=Path, default=base.ANSWER_PATH, help="テストの正解のCSV（評価用）"
    )
    parser.add_argument("--no-answer", action="store_true", help="正解との比較をしない")
    parser.add_argument("--no-tracking", action="store_true", help="MLflowに記録しない")
    parser.add_argument(
        "--output-root", type=Path, default=None, help="出力のルート（既定: outputs/experiments）"
    )
    args = parser.parse_args(argv)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = base.setup_logging(
        outputs_dir() / "logs" / f"train_predict_x50_regression_{args.model}_{stamp}.log"
    )
    logger.info(
        "引数: %s", {k: base.shown(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    )

    inputs = base.load_inputs(args.train, args.test, args.categories)
    config = make_config(
        args.model, len(inputs.categories), args.n_splits, args.n_trials, args.test_prediction
    )
    dataset, frame = prepare(config, inputs)
    tracker = base.make_tracker(not args.no_tracking)
    result = train_and_predict(config, dataset, tracker, args.output_root)
    answer = None if args.no_answer or not args.answer.is_file() else base.load_answer(args.answer)
    run = collect_predictions(inputs.categories, dataset, frame, inputs.test, result, answer)
    save_outputs(run, args.model, not args.no_tracking, log_path)
    logger.info("完了: %s", base.shown(result.output_dir))


if __name__ == "__main__":
    main()
