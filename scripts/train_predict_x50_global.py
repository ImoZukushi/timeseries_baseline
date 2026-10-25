"""前処理済みの x50 データ（long 型）で、全種類を1つのグローバルモデルで学習・再帰予測する。

`scripts/train_predict_x50.py`（種類ごとに別モデル）のグローバルモデル版。入力・リークの確認・
図の部品は同じものを使う。

- 入力は long 型（1行 = 日付 × 目的変数の種類）なので、そのまま使う。種類の列 `Y_cat` を系列IDにし、
  1つのモデルが全種類の行で学習する。
- モデルの特徴量は「その行の種類自身のラグ・移動平均・前日比 + 系列ID + 日付 + 説明変数」。
  ラグ等は系列ごとに作る（`forecast.series_col`）ため、他の種類の値は混ざらない。
  系列IDは `feature_engineering.categorical.PandasOrdinalEncoder` で整数にする
  （CVの fold ごとに学習期間で fit）。
- 目的変数は対数で学習・予測し（`forecast.target_transform="log"`）、評価・出力は元の尺度。
  対数にすると水準の違う種類も「変化率」の尺度で近くなり、1つのモデルで扱いやすくなる。
- 目的変数のラグ・移動平均・前日比は、検証・テスト期間では再帰予測の中で
  （全系列を1日ずつ同時に進めて）予測値から作り直す。学習データのCSVにある値と一致することを
  確かめる。
- CVは日付で区切る時系列CV（全系列で同じ境目）。検証期間も再帰予測で評価する。
- 正解ファイルは、テスト予測が終わった後の評価にだけ使う。
- 欠損の扱いは `train_predict_x50.py` と同じ（目的変数が欠損の行は学習・評価から除き、履歴としては
  残す。ラグ・移動平均・説明変数の欠損は埋めない）。

注意: 学習の早期終了・チューニングの指標（MAE）は全系列の行をまとめて計算するため、
水準の高い種類の誤差の影響が大きくなる。種類ごとの MAE・相対MAE は要約の表で確認する。

出力（`outputs/experiments/x50_global_{モデル名}/{実行日時}/`）:
    - `run_experiment` の出力一式（CVスコア・OOF/テスト予測・誤差評価の図（系列ごと）・SHAP・設定）
    - `summary.csv`（種類ごとの MAE）、`test_predictions.csv`（全種類のテスト予測。long 型）、
      `shap_importance_by_target.csv`
    - `figures/`: `summary_mae.png`・`training_overview.png`・`test_overview.png`・
      `shap_overview.png`
    - `run.log`。MLflow（`mlruns/`）にも指標・図・ログを記録する。

使い方:
    uv run python scripts/train_predict_x50_global.py --model lightgbm
    uv run python scripts/train_predict_x50_global.py --model xgboost --categories Y1 Y2
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import shutil
import sys
import time
from collections.abc import Sequence
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
from modeling.forecasting import ForecastCVResult, make_builder  # noqa: E402
from modeling.metrics import get_metric  # noqa: E402
from modeling.tracking import Tracker  # noqa: E402
from util.paths import ensure_parent_dir, outputs_dir  # noqa: E402
from util.plotting import add_caption, ensure_japanese_font  # noqa: E402

logger = logging.getLogger("train_predict_x50_global")


@dataclass
class GlobalRun:
    """グローバルモデルの学習・予測の結果。

    Attributes:
        categories: 目的変数の種類（系列）の並び。
        dataset: 学習用データ一式（long 型）。
        result: `run_experiment` の結果。
        predictions: テスト予測（long 型。列 `date`, `Y_cat`, `pred`。正解があれば `actual` も）。
        oof: 学習期間の OOF 予測
            （long 型。列 `date`, `Y_cat`, `actual`, `pred`, `onestep`, `fold`）。
    """

    categories: list[str]
    dataset: Dataset
    result: ExperimentResult
    predictions: pd.DataFrame
    oof: pd.DataFrame


# --- モデルに渡す表 ---------------------------------------------------------------------


def model_frames(inputs: base.Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`prepare_dataset` に渡す学習・テストの表を作る（long 型のまま、全種類の行を使う）。

    学習の表は「日付・系列ID（`Y_cat`）・日付の特徴量・説明変数・目的変数（元の尺度）」だけにする。
    CSVにある目的変数の特徴量は外し、`modeling` に系列ごとに作り直させる
    （テスト期間は再帰予測の中で作る）。行は日付の昇順、同じ日付の中は種類の順。

    Returns:
        (学習の表, テストの表（目的変数なし）)。
    """
    columns = [base.DATE_COL, base.CAT_COL, *inputs.exog_cols]
    train = inputs.train[columns].assign(
        **{base.TARGET_COL: inputs.train[f"{base.TARGET_COL}{base.RAW_SUFFIX}"]}
    )
    test = inputs.test[columns]
    assert base.TARGET_COL not in test.columns
    return train.reset_index(drop=True), test.reset_index(drop=True)


def csv_target_features(inputs: base.Inputs) -> pd.DataFrame:
    """学習データのCSVにある目的変数の特徴量（種類ごとに作ったもの）を、日付・種類つきで返す。"""
    return inputs.train[[base.DATE_COL, base.CAT_COL, *inputs.spec.columns]]


# --- 実験設定 ---------------------------------------------------------------------------


def make_config(
    model: str,
    spec: base.TargetFeatureSpec,
    n_series: int,
    cutoffs: Sequence[str] = base.DEFAULT_CUTOFFS,
    n_trials: int = 0,
    test_prediction: str = "refit_full",
) -> ExperimentConfig:
    """グローバルモデルの実験設定を作る（`ExperimentConfig` で検証する）。

    Args:
        model: モデル名（`lightgbm` / `xgboost`）。
        spec: 目的変数の特徴量の定義（全種類で共通）。
        n_series: 種類（系列）の数。誤差評価の図を全系列分描く。
        cutoffs: 時系列CVの検証期間の開始日。
        n_trials: Optunaの試行回数（0ならチューニングしない）。
        test_prediction: テスト予測の作り方（`refit_full` / `fold_mean`）。

    Raises:
        ValueError: 未対応のモデル名の場合。
    """
    if model not in base.MODELS:
        raise ValueError(f"モデルは {base.MODELS} から選んでください: {model}")
    return ExperimentConfig.model_validate(
        {
            "name": f"x50_global_{model}",
            "task": "time_series",
            "data": {
                "train_path": str(base.TRAIN_PATH),
                "test_path": str(base.TEST_PATH),
                "target": spec.target,
                # 学習データは日付の昇順に並べ替えられる（同じ日付の中は元の並び＝種類の順）
                "time_col": base.DATE_COL,
                "drop_cols": [base.DATE_COL],
            },
            # 系列ID（Y1〜Y10 の文字列）を整数にして、モデルが種類を区別できるようにする。
            # CVの各 fold の学習期間だけで fit される
            # （全種類が学習期間にあるので未知のIDは出ない）
            "features": [
                {
                    "class": "feature_engineering.categorical.PandasOrdinalEncoder",
                    "params": {"variables": [base.CAT_COL]},
                }
            ],
            # 日付で区切る時系列CV。全系列で同じ境目になる
            "cv": {
                "method": "time_cutoff",
                "time_column": base.DATE_COL,
                "cutoffs": list(cutoffs),
            },
            "model": {"name": model, "params": {"n_estimators": 1000}, "early_stopping_rounds": 50},
            "metrics": ["mae"],
            "test_prediction": test_prediction,
            "forecast": {
                # ラグ・移動平均・前日比は系列（種類）ごとに作る
                "series_col": base.CAT_COL,
                "lags": spec.lags,
                "rolling_windows": spec.rolling_windows,
                # 目的変数に欠損があっても、窓内の観測値だけで移動平均を作る
                "rolling_min_periods": base.ROLLING_MIN_PERIODS,
                "rate_of_change": spec.rate_of_change,
                "target_transform": "log",
            },
            "tuning": {"enabled": n_trials > 0, "n_trials": max(n_trials, 1)},
            "explain": {"enabled": True},
            "evaluation": {"enabled": True, "training_history": True, "max_series": n_series},
            "tracking": {"experiment_name": base.MLFLOW_EXPERIMENT},
        }
    )


# --- 学習と確認 -------------------------------------------------------------------------


def prepare(config: ExperimentConfig, inputs: base.Inputs) -> Dataset:
    """全種類の学習用データ一式を作り、リークのおそれがないことを確かめる。

    確かめること:
        - 系列ごとに作り直した目的変数の特徴量が、学習データのCSVの値（種類ごとに作ったもの）と
          一致する（＝他の種類の値が混ざっていない・過去の値だけから作られている）。
          各系列のCSVの先頭 `lookback` 日は、CSVに無い過去の日を使うため比べない。
          欠損の位置も一致することを確かめる。
        - モデルの特徴量に、目的変数（`Y`・`Y_raw`）の列が無い。
        - CVの各foldで、学習期間がすべて検証期間より前にある（全系列で同じ境目）。
        - テストデータに目的変数が無い。

    Raises:
        base.LeakageError: 上のいずれかを満たさない場合。
    """
    spec = inputs.spec
    train, test = model_frames(inputs)
    dataset = prepare_dataset(config, train, test)
    assert dataset.frame is not None

    keys = [base.DATE_COL, base.CAT_COL]
    rebuilt = pd.concat([dataset.frame[keys], dataset.X[spec.columns]], axis=1)
    merged = rebuilt.merge(
        csv_target_features(inputs),
        on=keys,
        how="left",
        suffixes=("", "_csv"),
        validate="1:1",
    )
    # 各系列のCSVの先頭 lookback 日は、CSVに無い過去の日を使うため作り直しても一致しない
    lookback = make_builder(config).lookback
    position = dataset.frame.groupby(base.CAT_COL, sort=False).cumcount().to_numpy()
    comparable = position >= lookback
    for col in spec.columns:
        if not np.allclose(
            merged.loc[comparable, col].to_numpy(dtype=float),
            merged.loc[comparable, f"{col}_csv"].to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise base.LeakageError(
                f"系列ごとに作り直した {col} が、学習データのCSVの値と一致しません"
            )
    raw_col = f"{base.TARGET_COL}{base.RAW_SUFFIX}"
    leaked = [c for c in dataset.X.columns if c in (base.TARGET_COL, raw_col)]
    if leaked:
        raise base.LeakageError(f"モデルの特徴量に目的変数の列があります: {leaked}")
    n_categories = len(inputs.categories)
    logger.info(
        "目的変数の特徴量: 系列ごとに作り直した値がCSVと一致"
        "（%d系列 × 先頭 %d日は過去の日数が足りないため比べない）",
        n_categories,
        lookback,
    )
    n_missing = int(np.isnan(dataset.y).sum())
    if n_missing:
        logger.info(
            "目的変数が欠損の %d行（%.1f%%）は学習・評価に使わない（履歴としては残す）",
            n_missing,
            100 * n_missing / len(dataset.y),
        )

    dates = pd.to_datetime(dataset.frame[base.DATE_COL])
    for i, (train_idx, valid_idx) in enumerate(dataset.folds):
        if dates.iloc[train_idx].max() >= dates.iloc[valid_idx].min():
            raise base.LeakageError(f"fold {i}: 学習期間が検証期間より後の日を含みます")
        n_series = dataset.frame[base.CAT_COL].iloc[valid_idx].nunique()
        logger.info(
            "fold %d: 学習 %s〜%s → 検証 %s〜%s（%d系列 × %d日、再帰予測で評価）",
            i,
            f"{dates.iloc[train_idx].min():%Y-%m-%d}",
            f"{dates.iloc[train_idx].max():%Y-%m-%d}",
            f"{dates.iloc[valid_idx].min():%Y-%m-%d}",
            f"{dates.iloc[valid_idx].max():%Y-%m-%d}",
            n_series,
            len(valid_idx) // n_series,
        )
    if dataset.test_frame is None or base.TARGET_COL in dataset.test_frame.columns:
        raise base.LeakageError(
            "テストデータに目的変数があります（またはテストデータがありません）"
        )
    logger.info(
        "学習データ（long 型）: %d行 = %d日 × %d系列,"
        " 特徴量 %d列（系列ID・自身のラグ等・日付・説明変数）",
        len(dataset.X),
        len(dataset.X) // n_categories,
        n_categories,
        dataset.X.shape[1],
    )
    return dataset


def train_and_predict(
    config: ExperimentConfig,
    dataset: Dataset,
    tracker: Tracker,
    output_root: Path | None = None,
    extra_tags: dict[str, str] | None = None,
) -> ExperimentResult:
    """全系列のCV学習（再帰バックテスト）・テスト予測・誤差評価・SHAP を `run_experiment` で行う。

    `run_experiment` の設定・ログの記録先は `train_predict_x50` と同じ。
    """
    logger.info(
        "学習開始: グローバルモデル=%s, チューニング=%s, テスト予測=%s",
        config.model.name,
        f"{config.tuning.n_trials}試行" if config.tuning.enabled else "なし",
        config.test_prediction,
    )
    start = time.perf_counter()
    result = run_experiment(
        config,
        dataset,
        tracker=tracker,
        output_root=output_root or outputs_dir() / "experiments",
        extra_tags={"model_type": "global", **(extra_tags or {})},
    )
    cv = result.cv_result
    logger.info(
        "学習終了（%.1f秒）: 全系列の CV MAE（再帰予測）fold別 %s / OOF %.4f。出力 %s",
        time.perf_counter() - start,
        [round(s, 4) for s in cv.fold_scores["mae"]],
        cv.oof_scores["mae"],
        base.shown(result.output_dir),
    )
    if result.tuning_result is not None:
        logger.info("チューニングの最良パラメータ: %s", result.tuning_result.best_params)
    return result


def collect_predictions(
    categories: list[str], dataset: Dataset, result: ExperimentResult, answer: pd.DataFrame | None
) -> GlobalRun:
    """テスト予測と OOF 予測を、日付・種類つきの long 型の表にする（正解があれば付ける）。"""
    assert dataset.frame is not None and dataset.test_frame is not None
    cv = result.cv_result
    assert cv.test_pred is not None
    keys = [base.DATE_COL, base.CAT_COL]
    predictions = pd.DataFrame(
        {
            base.DATE_COL: dataset.test_frame[base.DATE_COL].to_numpy(),
            base.CAT_COL: dataset.test_frame[base.CAT_COL].to_numpy(),
            base.PRED_COL: cv.test_pred,
        }
    )
    if answer is not None:
        # 正解（long 型）はテスト予測が終わった後にだけ使う
        actual = answer[[*keys, base.TARGET_COL]].rename(columns={base.TARGET_COL: base.ACTUAL_COL})
        # 正解の値が欠損の行は、評価（MAE）から除く（actual は欠損のまま残す）
        predictions = predictions.merge(actual, on=keys, how="left", validate="1:1", indicator=True)
        if (predictions["_merge"] != "both").any():
            raise ValueError("正解ファイルに、テスト期間の一部の日付・種類がありません")
        predictions = predictions.drop(columns="_merge")
    oof = pd.DataFrame(
        {
            base.DATE_COL: dataset.frame[base.DATE_COL].to_numpy(),
            base.CAT_COL: dataset.frame[base.CAT_COL].to_numpy(),
            base.ACTUAL_COL: dataset.y,
            base.PRED_COL: cv.oof_pred,
            "onestep": cv.onestep_oof_pred if isinstance(cv, ForecastCVResult) else np.nan,
            "fold": cv.fold_ids,
        }
    )
    return GlobalRun(categories, dataset, result, predictions, oof)


# --- 要約と図 ---------------------------------------------------------------------------


def summary_frame(run: GlobalRun) -> pd.DataFrame:
    """種類（系列）ごとの MAE をまとめた表（`train_predict_x50.summary_frame` と同じ列）。

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
        onestep = oof["onestep"].notna()
        if onestep.any():
            row["cv_onestep_mae"] = mae(
                oof.loc[onestep, actual_col].to_numpy(), oof.loc[onestep, "onestep"].to_numpy()
            )
        if actual_col in run.predictions.columns:
            test = run.predictions[run.predictions[base.CAT_COL] == category]
            test_mae = mae(test[actual_col].to_numpy(), test[pred_col].to_numpy())
            row["test_mae"] = test_mae
            row["test_relative_mae"] = test_mae / float(test[actual_col].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot_training_overview(run: GlobalRun, title: str, ncols: int = 2) -> plt.Figure:
    """学習結果: 系列ごとに、学習期間の実測値と検証期間の再帰予測（OOF）を並べて描く。"""
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
        predicted = part["fold"] >= 0
        for fold in sorted(part.loc[predicted, "fold"].unique()):
            f = part[part["fold"] == fold]
            ax.plot(f[date_col], f[pred_col], color=f"C{fold}", linewidth=1, label=f"fold {fold}")
        score = mae(
            part.loc[predicted, actual_col].to_numpy(), part.loc[predicted, pred_col].to_numpy()
        )
        ax.set_title(f"{category}（CV MAE={score:.2f}）")
    for ax in list(axes.flat)[len(run.categories) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(f"{title}: 学習期間の実測値と、検証期間の再帰予測（系列ごと）")
    add_caption(
        fig, "1つのモデルで全種類を学習。検証期間は実測値を使わず、全系列を同時に再帰予測した。"
    )
    return fig


def plot_test_overview(run: GlobalRun, title: str, ncols: int = 2) -> plt.Figure:
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
        ax.plot(part[date_col], part[pred_col], color="C3", linewidth=1.2, label="再帰予測")
        ax.set_title(label)
    for ax in list(axes.flat)[len(run.categories) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(f"{title}: テスト期間の予測（系列ごと）")
    return fig


def shap_importance_table(run: GlobalRun) -> pd.DataFrame:
    """系列ごとの SHAP 重要度（平均|SHAP値|の割合）を、特徴量 × 種類の表にする。

    グローバルモデルの OOF SHAP 値を、行の系列（種類）で分けて平均する。目的変数自身のラグ等
    （`Y_lag_1` など）は `自身_lag_1` に、系列ID（`Y_cat`）は `系列ID` に名前を変える
    （004 の図と揃える）。
    """
    assert run.dataset.frame is not None
    values = pd.read_parquet(run.result.output_dir / "shap" / "shap_values.parquet")
    series = run.dataset.frame[base.CAT_COL].to_numpy()[values.pop("row").to_numpy()]
    target = base.TARGET_COL
    rename = {
        c: (
            "系列ID"
            if c == base.CAT_COL
            else f"自身{c[len(target) :]}"
            if c.startswith(f"{target}_")
            else c
        )
        for c in values.columns
    }
    importance = values.abs().groupby(series).mean().T.rename(index=rename)
    importance = importance[[c for c in run.categories if c in importance.columns]]
    return importance / importance.sum()


def save_outputs(
    run: GlobalRun, model: str, tracker_enabled: bool, log_path: Path | None = None
) -> pd.DataFrame:
    """要約の表・予測・図・ログを実験の出力ディレクトリに保存し、MLflow の同じ run にも記録する。"""
    out = run.result.output_dir
    title = f"x50_global_{model}"
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
    """グローバルモデルの学習・テスト予測・評価・図・ログの保存を行う。"""
    import matplotlib

    # 図はファイルに保存するだけなので、画面を使わないバックエンドにする
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=base.MODELS, default="lightgbm", help="モデル")
    parser.add_argument(
        "--categories", nargs="+", default=None, help="目的変数の種類（既定: 全種類）"
    )
    parser.add_argument(
        "--n-trials", type=int, default=0, help="Optunaの試行回数（0ならチューニングしない）"
    )
    parser.add_argument(
        "--cutoffs", nargs="+", default=base.DEFAULT_CUTOFFS, help="CVの検証期間の開始日"
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
        outputs_dir() / "logs" / f"train_predict_x50_global_{args.model}_{stamp}.log"
    )
    logger.info(
        "引数: %s", {k: base.shown(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    )

    inputs = base.load_inputs(args.train, args.test, args.categories)
    config = make_config(
        args.model,
        inputs.spec,
        len(inputs.categories),
        args.cutoffs,
        args.n_trials,
        args.test_prediction,
    )
    dataset = prepare(config, inputs)
    tracker = base.make_tracker(not args.no_tracking)
    result = train_and_predict(config, dataset, tracker, args.output_root)
    answer = None if args.no_answer or not args.answer.is_file() else base.load_answer(args.answer)
    run = collect_predictions(inputs.categories, dataset, result, answer)
    save_outputs(run, args.model, not args.no_tracking, log_path)
    logger.info("完了: %s", base.shown(result.output_dir))


if __name__ == "__main__":
    main()
