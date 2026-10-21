"""前処理済みの x50 データで、目的変数 Y1〜Y10 を1つのグローバルモデルで学習・再帰予測する。

`scripts/train_predict_x50.py`（目的変数ごとに別モデル）のグローバルモデル版。入力・リークの確認・
図の部品は同じものを使う。

- 学習・テストのデータを縦持ち（1行 = 日付 × 目的変数）に変換し、目的変数の種類を系列ID
  （列 `series`）にする。1つのモデルが全目的変数の行で学習する。
- モデルの特徴量は「その行の目的変数自身のラグ・移動平均・前日比 + 系列ID + 日付 + 説明変数」。
  ラグ等は系列ごとに作る（`forecast.series_col`）ため、他の目的変数の値は混ざらない。
  系列IDは `feature_engineering.categorical.PandasOrdinalEncoder` で整数にする
  （CVの fold ごとに学習期間で fit）。
- 目的変数は対数で学習・予測し（`forecast.target_transform="log"`）、評価・出力は元の尺度。
  対数にすると水準の違う目的変数も「変化率」の尺度で近くなり、1つのモデルで扱いやすくなる。
- 目的変数のラグ・移動平均・前日比は、検証・テスト期間では再帰予測の中で
  （全系列を1日ずつ同時に進めて）予測値から作り直す。学習データのCSVにある値と一致することを
  確かめる。
- CVは日付で区切る時系列CV（全系列で同じ境目）。検証期間も再帰予測で評価する。
- 正解ファイルは、テスト予測が終わった後の評価にだけ使う。

注意: 学習の早期終了・チューニングの指標（MAE）は全系列の行をまとめて計算するため、
水準の高い目的変数の誤差の影響が大きくなる。目的変数ごとの MAE・相対MAE は要約の表で確認する。

出力（`outputs/experiments/x50_global_{モデル名}/{実行日時}/`）:
    - `run_experiment` の出力一式（CVスコア・OOF/テスト予測・誤差評価の図（系列ごと）・SHAP・設定）
    - `summary.csv`（目的変数ごとの MAE）、`test_predictions.csv`（全目的変数のテスト予測）、
      `shap_importance_by_target.csv`
    - `figures/`: `summary_mae.png`・`training_overview.png`・`test_overview.png`・
      `shap_overview.png`
    - `run.log`。MLflow（`mlruns/`）にも指標・図・ログを記録する。

使い方:
    uv run python scripts/train_predict_x50_global.py --model lightgbm
    uv run python scripts/train_predict_x50_global.py --model xgboost --targets Y1 Y2 --n-trials 20
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
from modeling.forecasting import ForecastCVResult  # noqa: E402
from modeling.metrics import get_metric  # noqa: E402
from modeling.tracking import Tracker  # noqa: E402
from util.paths import ensure_parent_dir, outputs_dir  # noqa: E402
from util.plotting import add_caption, ensure_japanese_font  # noqa: E402

DATE_COL = base.DATE_COL
SERIES_COL = "series"  # 縦持ちの表で、目的変数の種類（Y1〜Y10）を表す系列ID
VALUE_COL = "value"  # 縦持ちの表の目的変数（元の尺度）
PRED_COL = base.PRED_COL
ACTUAL_COL = base.ACTUAL_COL

logger = logging.getLogger("train_predict_x50_global")


@dataclass
class GlobalRun:
    """グローバルモデルの学習・予測の結果。

    Attributes:
        targets: 目的変数（系列）の並び。
        dataset: 学習用データ一式（縦持ち）。
        result: `run_experiment` の結果。
        predictions: テスト予測（縦持ち。列 `date`, `series`, `pred`。正解があれば `actual` も）。
        oof: 学習期間の OOF 予測
            （縦持ち。列 `date`, `series`, `actual`, `pred`, `onestep`, `fold`）。
    """

    targets: list[str]
    dataset: Dataset
    result: ExperimentResult
    predictions: pd.DataFrame
    oof: pd.DataFrame


# --- 縦持ちへの変換 ---------------------------------------------------------------------


def common_spec(inputs: base.Inputs) -> base.TargetFeatureSpec:
    """全目的変数で共通の、目的変数の特徴量の定義を返す（列名は `value_lag_1` など）。

    Raises:
        ValueError: 目的変数によって特徴量の定義（ラグ・窓幅）が違う場合（1つのモデルで扱えない）。
    """
    specs = [inputs.specs[t] for t in inputs.targets]
    first = specs[0]
    for spec in specs[1:]:
        if (spec.lags, spec.rolling_windows, spec.rate_of_change) != (
            first.lags,
            first.rolling_windows,
            first.rate_of_change,
        ):
            raise ValueError(f"{spec.target} の特徴量の定義が {first.target} と違います")
    columns = [f"{VALUE_COL}{c[len(first.target) :]}" for c in first.columns]
    return base.TargetFeatureSpec(
        VALUE_COL, first.lags, first.rolling_windows, first.rate_of_change, columns
    )


def to_long(inputs: base.Inputs) -> tuple[pd.DataFrame, pd.DataFrame]:
    """学習・テストのデータを縦持ち（1行 = 日付 × 目的変数）にする。

    学習の表は「日付・系列ID・日付の特徴量・説明変数・目的変数（元の尺度）」だけにする。
    CSVにある目的変数の特徴量は外し、`modeling` に系列ごとに作り直させる
    （テスト期間は再帰予測の中で作る）。
    行は日付の昇順、同じ日付の中は目的変数の並び順。

    Returns:
        (学習の表, テストの表（目的変数なし）)。
    """
    train_parts, test_parts = [], []
    for target in inputs.targets:
        part = inputs.train[[DATE_COL, *inputs.exog_cols]].copy()
        part[SERIES_COL] = target
        part[VALUE_COL] = inputs.train[f"{target}{base.RAW_SUFFIX}"]
        train_parts.append(part)
        test_parts.append(inputs.test[[DATE_COL, *inputs.exog_cols]].assign(**{SERIES_COL: target}))
    columns = [DATE_COL, SERIES_COL, *inputs.exog_cols]
    train = pd.concat(train_parts, ignore_index=True)[[*columns, VALUE_COL]]
    test = pd.concat(test_parts, ignore_index=True)[columns]
    order = {t: i for i, t in enumerate(inputs.targets)}
    keys = [DATE_COL, SERIES_COL]
    train = train.sort_values(keys, key=lambda s: s.map(order) if s.name == SERIES_COL else s)
    test = test.sort_values(keys, key=lambda s: s.map(order) if s.name == SERIES_COL else s)
    assert VALUE_COL not in test.columns
    return train.reset_index(drop=True), test.reset_index(drop=True)


def csv_target_features(inputs: base.Inputs, spec: base.TargetFeatureSpec) -> pd.DataFrame:
    """学習データのCSVにある目的変数の特徴量を縦持ちにする（列名は `value_lag_1` などに揃える）。"""
    parts = []
    for target in inputs.targets:
        own = inputs.specs[target].columns
        part = inputs.train[[DATE_COL, *own]].rename(
            columns=dict(zip(own, spec.columns, strict=True))
        )
        parts.append(part.assign(**{SERIES_COL: target}))
    return pd.concat(parts, ignore_index=True)


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
        spec: 目的変数の特徴量の定義（全目的変数で共通）。
        n_series: 目的変数（系列）の数。誤差評価の図を全系列分描く。
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
                "target": VALUE_COL,
                # 学習データは日付の昇順に並べ替えられる（同じ日付の中は元の並び＝目的変数の順）
                "time_col": DATE_COL,
                "drop_cols": [DATE_COL],
            },
            # 系列ID（Y1〜Y10 の文字列）を整数にして、モデルが目的変数を区別できるようにする。
            # CVの各 fold の学習期間だけで fit される
            # （全目的変数が学習期間にあるので未知のIDは出ない）
            "features": [
                {
                    "class": "feature_engineering.categorical.PandasOrdinalEncoder",
                    "params": {"variables": [SERIES_COL]},
                }
            ],
            # 日付で区切る時系列CV。全系列で同じ境目になる
            "cv": {"method": "time_cutoff", "time_column": DATE_COL, "cutoffs": list(cutoffs)},
            "model": {"name": model, "params": {"n_estimators": 1000}, "early_stopping_rounds": 50},
            "metrics": ["mae"],
            "test_prediction": test_prediction,
            "forecast": {
                # ラグ・移動平均・前日比は系列（目的変数）ごとに作る
                "series_col": SERIES_COL,
                "lags": spec.lags,
                "rolling_windows": spec.rolling_windows,
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


def prepare(config: ExperimentConfig, inputs: base.Inputs, spec: base.TargetFeatureSpec) -> Dataset:
    """縦持ちの学習用データ一式を作り、リークのおそれがないことを確かめる。

    確かめること:
        - 系列ごとに作り直した目的変数の特徴量が、学習データのCSVの値（目的変数ごとに作ったもの）と
          一致する（＝他の目的変数の値が混ざっていない・過去の値だけから作られている）。
        - モデルの特徴量に、元の目的変数の列（`Yk` など）が無い。
        - CVの各foldで、学習期間がすべて検証期間より前にある（全系列で同じ境目）。
        - テストデータに目的変数が無い。

    Raises:
        base.LeakageError: 上のいずれかを満たさない場合。
    """
    train, test = to_long(inputs)
    dataset = prepare_dataset(config, train, test)
    assert dataset.frame is not None

    keys = [DATE_COL, SERIES_COL]
    rebuilt = pd.concat([dataset.frame[keys], dataset.X[spec.columns]], axis=1)
    merged = rebuilt.merge(
        csv_target_features(inputs, spec),
        on=keys,
        how="left",
        suffixes=("", "_csv"),
        validate="1:1",
    )
    for col in spec.columns:
        comparable = merged[col].notna()
        if not np.allclose(merged.loc[comparable, col], merged.loc[comparable, f"{col}_csv"]):
            raise base.LeakageError(
                f"系列ごとに作り直した {col} が、学習データのCSVの値と一致しません"
            )
    leaked = [
        c for c in dataset.X.columns if c in inputs.targets or c.startswith(tuple(inputs.targets))
    ]
    if leaked:
        raise base.LeakageError(f"モデルの特徴量に元の目的変数の列があります: {leaked}")
    n_head = int(merged[spec.columns].isna().any(axis=1).sum())
    logger.info(
        "目的変数の特徴量: 系列ごとに作り直した値がCSVと一致"
        "（%d系列 × 先頭 %d日は過去の日数が足りないため欠損）",
        len(inputs.targets),
        n_head // len(inputs.targets),
    )

    dates = pd.to_datetime(dataset.frame[DATE_COL])
    for i, (train_idx, valid_idx) in enumerate(dataset.folds):
        if dates.iloc[train_idx].max() >= dates.iloc[valid_idx].min():
            raise base.LeakageError(f"fold {i}: 学習期間が検証期間より後の日を含みます")
        n_series = dataset.frame[SERIES_COL].iloc[valid_idx].nunique()
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
    if dataset.test_frame is None or VALUE_COL in dataset.test_frame.columns:
        raise base.LeakageError(
            "テストデータに目的変数があります（またはテストデータがありません）"
        )
    logger.info(
        "学習データ（縦持ち）: %d行 = %d日 × %d系列,"
        " 特徴量 %d列（系列ID・自身のラグ等・日付・説明変数）",
        len(dataset.X),
        len(dataset.X) // len(inputs.targets),
        len(inputs.targets),
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
    targets: list[str], dataset: Dataset, result: ExperimentResult, answer: pd.DataFrame | None
) -> GlobalRun:
    """テスト予測と OOF 予測を、日付・系列つきの縦持ちの表にする（正解があれば付ける）。"""
    assert dataset.frame is not None and dataset.test_frame is not None
    cv = result.cv_result
    assert cv.test_pred is not None
    predictions = pd.DataFrame(
        {
            DATE_COL: dataset.test_frame[DATE_COL].to_numpy(),
            SERIES_COL: dataset.test_frame[SERIES_COL].to_numpy(),
            PRED_COL: cv.test_pred,
        }
    )
    if answer is not None:
        # 正解はテスト予測が終わった後にだけ使う
        actual = answer.melt(
            id_vars=DATE_COL, value_vars=targets, var_name=SERIES_COL, value_name=ACTUAL_COL
        )
        predictions = predictions.merge(
            actual, on=[DATE_COL, SERIES_COL], how="left", validate="1:1"
        )
        if predictions[ACTUAL_COL].isna().any():
            raise ValueError("正解ファイルに、テスト期間の一部の日付・目的変数がありません")
    oof = pd.DataFrame(
        {
            DATE_COL: dataset.frame[DATE_COL].to_numpy(),
            SERIES_COL: dataset.frame[SERIES_COL].to_numpy(),
            ACTUAL_COL: dataset.y,
            PRED_COL: cv.oof_pred,
            "onestep": cv.onestep_oof_pred if isinstance(cv, ForecastCVResult) else np.nan,
            "fold": cv.fold_ids,
        }
    )
    return GlobalRun(targets, dataset, result, predictions, oof)


# --- 要約と図 ---------------------------------------------------------------------------


def summary_frame(run: GlobalRun) -> pd.DataFrame:
    """目的変数（系列）ごとの MAE をまとめた表（`train_predict_x50.summary_frame` と同じ列）。"""
    mae = get_metric("mae")
    rows = []
    for target in run.targets:
        oof = run.oof[(run.oof[SERIES_COL] == target) & (run.oof["fold"] >= 0)]
        row: dict[str, object] = {
            "target": target,
            "cv_oof_mae": mae(oof[ACTUAL_COL].to_numpy(), oof[PRED_COL].to_numpy()),
        }
        row["cv_relative_mae"] = float(row["cv_oof_mae"]) / float(oof[ACTUAL_COL].mean())  # type: ignore[arg-type]
        onestep = oof["onestep"].notna()
        if onestep.any():
            row["cv_onestep_mae"] = mae(
                oof.loc[onestep, ACTUAL_COL].to_numpy(), oof.loc[onestep, "onestep"].to_numpy()
            )
        if ACTUAL_COL in run.predictions.columns:
            test = run.predictions[run.predictions[SERIES_COL] == target]
            row["test_mae"] = mae(test[ACTUAL_COL].to_numpy(), test[PRED_COL].to_numpy())
            row["test_relative_mae"] = float(row["test_mae"]) / float(test[ACTUAL_COL].mean())  # type: ignore[arg-type]
        rows.append(row)
    return pd.DataFrame(rows)


def wide_predictions(run: GlobalRun) -> pd.DataFrame:
    """全目的変数のテスト予測を横に並べた表（列 `date`, `Yk_pred`, 正解があれば `Yk_actual`）。"""
    values = [PRED_COL] + ([ACTUAL_COL] if ACTUAL_COL in run.predictions.columns else [])
    wide = run.predictions.pivot(index=DATE_COL, columns=SERIES_COL, values=values)
    columns = [(v, t) for t in run.targets for v in values]
    wide = wide[columns]
    wide.columns = [f"{t}_{v}" for v, t in columns]
    return wide.reset_index()


def plot_training_overview(run: GlobalRun, title: str, ncols: int = 2) -> plt.Figure:
    """学習結果: 系列ごとに、学習期間の実測値と検証期間の再帰予測（OOF）を並べて描く。"""
    ensure_japanese_font()
    nrows = int(np.ceil(len(run.targets) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 2.6 * nrows), constrained_layout=True, sharex=True, squeeze=False
    )
    mae = get_metric("mae")
    for ax, target in zip(axes.flat, run.targets, strict=False):
        part = run.oof[run.oof[SERIES_COL] == target]
        ax.plot(part[DATE_COL], part[ACTUAL_COL], color="black", linewidth=0.7, label="実測")
        predicted = part["fold"] >= 0
        for fold in sorted(part.loc[predicted, "fold"].unique()):
            f = part[part["fold"] == fold]
            ax.plot(f[DATE_COL], f[PRED_COL], color=f"C{fold}", linewidth=1, label=f"fold {fold}")
        score = mae(
            part.loc[predicted, ACTUAL_COL].to_numpy(), part.loc[predicted, PRED_COL].to_numpy()
        )
        ax.set_title(f"{target}（CV MAE={score:.2f}）")
    for ax in list(axes.flat)[len(run.targets) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(f"{title}: 学習期間の実測値と、検証期間の再帰予測（系列ごと）")
    add_caption(
        fig, "1つのモデルで全目的変数を学習。検証期間は実測値を使わず、全系列を同時に再帰予測した。"
    )
    return fig


def plot_test_overview(run: GlobalRun, title: str, ncols: int = 2) -> plt.Figure:
    """テスト結果: 系列ごとに、テスト期間の予測（正解があれば実測も）を並べて描く。"""
    ensure_japanese_font()
    nrows = int(np.ceil(len(run.targets) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 2.6 * nrows), constrained_layout=True, sharex=True, squeeze=False
    )
    has_answer = ACTUAL_COL in run.predictions.columns
    mae = get_metric("mae")
    for ax, target in zip(axes.flat, run.targets, strict=False):
        part = run.predictions[run.predictions[SERIES_COL] == target]
        label = target
        if has_answer:
            ax.plot(part[DATE_COL], part[ACTUAL_COL], color="black", linewidth=0.8, label="実測")
            label += f"（MAE={mae(part[ACTUAL_COL].to_numpy(), part[PRED_COL].to_numpy()):.2f}）"
        ax.plot(part[DATE_COL], part[PRED_COL], color="C3", linewidth=1.2, label="再帰予測")
        ax.set_title(label)
    for ax in list(axes.flat)[len(run.targets) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(f"{title}: テスト期間の予測（系列ごと）")
    return fig


def shap_importance_table(run: GlobalRun) -> pd.DataFrame:
    """系列ごとの SHAP 重要度（平均|SHAP値|の割合）を、特徴量 × 目的変数の表にする。

    グローバルモデルの OOF SHAP 値を、行の系列（目的変数）で分けて平均する。目的変数自身のラグ等
    （`value_lag_1` など）は `自身_lag_1` に、系列ID は `系列ID` に名前を変える
    （004 の図と揃える）。
    """
    assert run.dataset.frame is not None
    values = pd.read_parquet(run.result.output_dir / "shap" / "shap_values.parquet")
    series = run.dataset.frame[SERIES_COL].to_numpy()[values.pop("row").to_numpy()]
    rename = {
        c: (
            "系列ID"
            if c == SERIES_COL
            else f"自身{c[len(VALUE_COL) :]}"
            if c.startswith(f"{VALUE_COL}_")
            else c
        )
        for c in values.columns
    }
    importance = values.abs().groupby(series).mean().T.rename(index=rename)
    importance = importance[[t for t in run.targets if t in importance.columns]]
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
    wide_predictions(run).to_csv(paths[-1], index=False, date_format="%Y-%m-%d")
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
    logger.info("%s（%d目的変数）。表: %s", message, len(run.targets), base.shown(paths[0]))
    if log_path is not None:
        for handler in logging.getLogger().handlers:
            handler.flush()
        paths.append(Path(shutil.copy(log_path, out / "run.log")))

    # run_experiment のrun（終了済み）に、目的変数ごとの指標と追加の出力を記録する
    if tracker_enabled and run.result.run_id is not None:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        metrics = {"mean_cv_relative_mae": float(summary["cv_relative_mae"].mean())}
        for row in summary.to_dict("records"):
            metrics[f"cv_oof_mae_{row['target']}"] = float(row["cv_oof_mae"])
            if "test_mae" in row:
                metrics[f"test_mae_{row['target']}"] = float(row["test_mae"])
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
    parser.add_argument("--targets", nargs="+", default=None, help="目的変数（既定: 全目的変数）")
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

    inputs = base.load_inputs(args.train, args.test, args.targets)
    spec = common_spec(inputs)
    config = make_config(
        args.model, spec, len(inputs.targets), args.cutoffs, args.n_trials, args.test_prediction
    )
    dataset = prepare(config, inputs, spec)
    tracker = base.make_tracker(not args.no_tracking)
    result = train_and_predict(config, dataset, tracker, args.output_root)
    answer = None if args.no_answer or not args.answer.is_file() else base.load_answer(args.answer)
    run = collect_predictions(inputs.targets, dataset, result, answer)
    save_outputs(run, args.model, not args.no_tracking, log_path)
    logger.info("完了: %s", base.shown(result.output_dir))


if __name__ == "__main__":
    main()
