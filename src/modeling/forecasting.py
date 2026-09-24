"""時系列の再帰的多段予測（recursive multi-step forecasting）。

学習は通常どおり「真の過去値から作ったラグ特徴量で1期先を予測する」モデルとして行い、
予測時は **予測値を次の時点の目的変数として履歴に追加し、ラグ等を再計算しながら
1ステップずつ進む**。これにより、実測値が得られない数期先まで予測できる。

目的変数から作る特徴量（ラグ・移動平均・前期比）は、過去の予測値を参照する必要があるため
sklearn Pipelineの外側（`TargetFeatureBuilder`）で系列ごとの履歴から作る。
外生変数の特徴量は従来どおり `ExperimentConfig.features`（Pipeline内）で作る。

評価（`recursive_backtest`）では、各foldの検証期間の目的変数を一切使わずに再帰予測する。
真のラグを使う1期先評価（`run_cv`）は長期予測の誤差を過小評価するため、
こちらの結果をOOF・チューニング・アンサンブルの基準にする。

前提:
    - 各系列の行は一定間隔（1行 = 1ステップ）であること。
    - 予測対象行の外生変数は予測時点で既知であること（カレンダー・予報値など）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline

from feature_engineering.time_series import (
    LagFeatureGenerator,
    MovingAverageTransformer,
    RateOfChangeTransformer,
)
from modeling.config import ExperimentConfig, ForecastConfig
from modeling.cv import Fold, folds_to_ids
from modeling.metrics import get_metric
from modeling.models import get_model_spec
from modeling.pipeline import MODEL_STEP
from modeling.tasks import Task, predict
from modeling.trainer import CVResult, FoldCallback, fit_pipeline

# 行の元の並びを復元するための一時列
_ROW = "__row__"
_IS_FUTURE = "__is_future__"


class TargetFeatureBuilder:
    """目的変数の履歴から特徴量（ラグ・移動平均・前期比）を作る。

    生成される列:
        - `{target}_lag_{k}`（`lags` の各k）
        - `{target}_lag_1_ma_{w}`（`rolling_windows` の各w。t-1 から過去w期の平均）
        - `{target}_lag_1_roc_1`（`rate_of_change` 指定時。(y_{t-1} - y_{t-2}) / y_{t-2}）

    いずれも t-1 以前の値だけを使うため、行 t の特徴量に y_t は含まれない。

    Args:
        target: 目的変数の列名。
        time_col: 時刻列名。
        config: 再帰予測の設定。
    """

    def __init__(self, target: str, time_col: str, config: ForecastConfig) -> None:
        self.target = target
        self.time_col = time_col
        self.config = config
        self.series_col = config.series_col
        lag1 = f"{target}_lag_1"
        group_by = self.series_col
        self._steps: list[Any] = [LagFeatureGenerator(target, config.lags, group_by=group_by)]
        self._steps += [
            MovingAverageTransformer(lag1, w, group_by=group_by) for w in config.rolling_windows
        ]
        if config.rate_of_change:
            self._steps.append(RateOfChangeTransformer(lag1, 1, group_by=group_by))
        for step in self._steps:
            step.fit(pl.DataFrame())

    @property
    def feature_names(self) -> list[str]:
        """生成する特徴量の列名。"""
        lag1 = f"{self.target}_lag_1"
        names = [f"{self.target}_lag_{k}" for k in self.config.lags]
        names += [f"{lag1}_ma_{w}" for w in self.config.rolling_windows]
        if self.config.rate_of_change:
            names.append(f"{lag1}_roc_1")
        return names

    @property
    def lookback(self) -> int:
        """特徴量の計算に必要な過去の行数（予測時は各系列の直近この行数だけを使う）。"""
        needs = [*self.config.lags, *self.config.rolling_windows]
        if self.config.rate_of_change:
            needs.append(2)
        return max(needs)

    def sort_keys(self) -> list[str]:
        """系列・時刻の並べ替えキー。"""
        return [self.time_col] if self.series_col is None else [self.series_col, self.time_col]

    def build(self, frame: pl.DataFrame) -> pl.DataFrame:
        """特徴量を追加したDataFrameを返す（行の並びは入力のまま）。"""
        indexed = frame.with_row_index(_ROW).sort(self.sort_keys(), maintain_order=True)
        for step in self._steps:
            indexed = step.transform(indexed)
        return indexed.sort(_ROW).drop(_ROW)


def step_index(frame: pl.DataFrame, time_col: str, series_col: str | None) -> np.ndarray:
    """各行が予測開始から何ステップ目か（系列内の時刻の順位、1始まり）を返す。"""
    rank = pl.col(time_col).rank(method="ordinal")
    if series_col is not None:
        rank = rank.over(series_col)
    return frame.select(rank).to_series().to_numpy().astype(np.int64)


def _check_future_after_history(
    history: pl.DataFrame, future: pl.DataFrame, time_col: str, series_col: str | None
) -> None:
    """各系列で、予測対象の時刻が履歴の最終時刻より後であることを確認する。

    Raises:
        ValueError: 予測対象が履歴と同時刻以前の系列がある場合。
    """
    keys = [] if series_col is None else [series_col]
    if history.height == 0 or future.height == 0:
        return
    if keys:
        last = history.group_by(keys).agg(pl.col(time_col).max().alias("_last"))
        first = future.group_by(keys).agg(pl.col(time_col).min().alias("_first"))
        joined = first.join(last, on=keys, how="inner")
    else:
        joined = pl.DataFrame(
            {"_first": [future[time_col].min()], "_last": [history[time_col].max()]}
        )
    bad = joined.filter(pl.col("_first") <= pl.col("_last"))
    if bad.height > 0:
        raise ValueError(
            "再帰予測では、予測対象の時刻が各系列の履歴（学習期間）より後である必要があります。"
            f"違反: {bad.head(5).to_dicts()}"
        )


def recursive_forecast(
    pipeline: Pipeline,
    history: pl.DataFrame,
    future: pl.DataFrame,
    builder: TargetFeatureBuilder,
    feature_columns: Sequence[str],
) -> np.ndarray:
    """予測値を履歴に追加しながら、未来の行を1ステップずつ予測する。

    各ステップでは全系列の「h番目の未来行」をまとめて予測する（系列ごとのループはしない）。

    Args:
        pipeline: 1期先予測用に学習済みのPipeline。
        history: 目的変数の実測値を含む過去の行。
        future: 予測対象の行（目的変数の値は参照しない）。
        builder: 目的変数の特徴量を作るビルダー。
        feature_columns: モデルに渡す列（学習時の `X` の列と同じ並び）。

    Returns:
        `future` の行順に並んだ予測値。
    """
    target, time_col, series_col = builder.target, builder.time_col, builder.series_col
    _check_future_after_history(history, future, time_col, series_col)
    clip = builder.config.clip
    # 予測対象行の目的変数は（あっても）使わないよう必ず空にする
    steps = step_index(future, time_col, series_col)
    fut = future.with_row_index(_ROW).with_columns(
        pl.lit(None, dtype=pl.Float64).alias(target), pl.Series("_step", steps)
    )
    # テストデータに無い学習専用の列（別の目的変数など）は使わない
    base_columns = [c for c in history.columns if c in fut.columns]
    hist = history.select(base_columns).with_columns(pl.col(target).cast(pl.Float64))
    preds = np.full(future.height, np.nan)

    for h in range(1, int(steps.max(initial=0)) + 1):
        current = fut.filter(pl.col("_step") == h)
        # 特徴量の計算に必要な直近 lookback 行だけを系列ごとに取り出す（計算量の削減）
        ordered = hist.sort(builder.sort_keys(), maintain_order=True)
        if series_col is None:
            tail = ordered.tail(builder.lookback)
        else:
            tail = ordered.group_by(series_col, maintain_order=True).tail(builder.lookback)
        tail = tail.select(base_columns)
        context = pl.concat(
            [
                tail.with_columns(pl.lit(False).alias(_IS_FUTURE)),
                current.select(base_columns).with_columns(pl.lit(True).alias(_IS_FUTURE)),
            ],
            how="vertical_relaxed",
        )
        built = builder.build(context).filter(pl.col(_IS_FUTURE))
        pred = predict(pipeline, built.select(list(feature_columns)), Task.REGRESSION)
        pred = np.clip(
            pred,
            -np.inf if clip.min is None else clip.min,
            np.inf if clip.max is None else clip.max,
        )
        preds[current[_ROW].to_numpy()] = pred
        # 予測値を目的変数として履歴に追加し、次のステップのラグの元にする
        hist = pl.concat(
            [hist, current.select(base_columns).with_columns(pl.Series(target, pred))],
            how="vertical_relaxed",
        )
    return preds


@dataclass
class ForecastCVResult(CVResult):
    """再帰バックテストの結果。

    `oof_pred` / `fold_scores` / `oof_scores` は再帰予測（検証期間の実測値を使わない）の値。

    Attributes:
        steps: 各行の予測ステップ（検証期間の何ステップ目か。予測しない行は0）。
        horizon_scores: ステップごとの指標（列: `step`, `n`, 各指標）。
        onestep_oof_pred: 参考: 真のラグを使った1期先予測のOOF。
        onestep_oof_scores: 参考: 1期先予測のOOFスコア。
    """

    steps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    horizon_scores: pl.DataFrame = field(default_factory=pl.DataFrame)
    onestep_oof_pred: np.ndarray = field(default_factory=lambda: np.zeros(0))
    onestep_oof_scores: dict[str, float] = field(default_factory=dict)


def make_builder(config: ExperimentConfig) -> TargetFeatureBuilder:
    """実験設定から `TargetFeatureBuilder` を作る。

    Raises:
        ValueError: `forecast` または `data.time_col` が未設定の場合。
    """
    if config.forecast is None or config.data.time_col is None:
        raise ValueError("再帰予測には forecast と data.time_col の設定が必要です")
    return TargetFeatureBuilder(config.data.target, config.data.time_col, config.forecast)


def recursive_backtest(
    config: ExperimentConfig,
    frame: pl.DataFrame,
    X: pl.DataFrame,
    y: np.ndarray,
    folds: Sequence[Fold],
    test_frame: pl.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    fold_callback: FoldCallback | None = None,
) -> ForecastCVResult:
    """fold毎に1期先モデルを学習し、検証期間を再帰予測して評価する。

    Args:
        config: 実験設定（`forecast` 必須）。
        frame: 学習データ全体（目的変数・時刻・系列列を含む。`X` と同じ行順）。
        X: 学習データの特徴量（目的変数の特徴量を含む。`frame` と同じ行順）。
        y: 目的変数。
        folds: CV分割。
        test_frame: 予測対象の将来の行（任意）。学習データ全体を履歴として再帰予測する。
        params: モデルパラメータ（Noneなら設定値）。
        fold_callback: fold完了ごとに `(fold番号, 再帰予測の主指標)` で呼ばれる関数。

    Returns:
        再帰バックテストの結果。
    """
    builder = make_builder(config)
    assert config.forecast is not None
    horizon = config.forecast.horizon
    time_col, series_col = builder.time_col, builder.series_col
    n = X.height
    metrics = [get_metric(name) for name in config.metrics]
    spec = get_model_spec(config.model.name)
    use_params = config.model.params if params is None else params
    feature_columns = X.columns

    oof = np.full(n, np.nan)
    onestep = np.full(n, np.nan)
    steps = np.zeros(n, dtype=np.int64)
    fold_scores: dict[str, list[float]] = {m.name: [] for m in metrics}
    models: list[Pipeline] = []
    best_iterations: list[int | None] = []
    evaluated_folds: list[Fold] = []

    for i, (train_idx, valid_idx) in enumerate(folds):
        pipeline = fit_pipeline(
            config, X[train_idx], y[train_idx], X[valid_idx], y[valid_idx], params=use_params
        )
        onestep[valid_idx] = predict(pipeline, X[valid_idx], Task.REGRESSION)

        valid_frame = frame[valid_idx]
        valid_steps = step_index(valid_frame, time_col, series_col)
        keep = valid_steps <= horizon if horizon is not None else np.ones(len(valid_idx), bool)
        target_idx = valid_idx[keep]
        pred = recursive_forecast(
            pipeline, frame[train_idx], frame[target_idx], builder, feature_columns
        )
        oof[target_idx] = pred
        steps[target_idx] = valid_steps[keep]
        for m in metrics:
            fold_scores[m.name].append(m(y[target_idx], pred))
        models.append(pipeline)
        best_iterations.append(spec.best_iteration(pipeline.named_steps[MODEL_STEP]))
        evaluated_folds.append((train_idx, target_idx))
        if fold_callback is not None:
            fold_callback(i, fold_scores[config.primary_metric][-1])

    # horizonで打ち切った行は「予測していない行」としてfold番号を-1にする
    fold_ids = folds_to_ids(evaluated_folds, n)
    predicted = fold_ids >= 0
    onestep_predicted = folds_to_ids(folds, n) >= 0

    test_pred: np.ndarray | None = None
    if test_frame is not None:
        if config.test_prediction == "fold_mean":
            test_pred = np.mean(
                [
                    recursive_forecast(m, frame, test_frame, builder, feature_columns)
                    for m in models
                ],
                axis=0,
            )
        else:
            full_params = use_params
            iters = [b for b in best_iterations if b is not None]
            if iters:
                full_params = spec.with_n_iterations(use_params, int(np.mean(iters)))
            full = fit_pipeline(config, X, y, params=full_params, early_stopping=False)
            models.append(full)
            test_pred = recursive_forecast(full, frame, test_frame, builder, feature_columns)

    return ForecastCVResult(
        oof_pred=oof,
        fold_ids=fold_ids,
        fold_scores=fold_scores,
        oof_scores={m.name: m(y[predicted], oof[predicted]) for m in metrics},
        test_pred=test_pred,
        models=models,
        best_iterations=best_iterations,
        params=dict(use_params),
        steps=steps,
        horizon_scores=_horizon_scores(y, oof, steps, [m.name for m in metrics]),
        onestep_oof_pred=onestep,
        onestep_oof_scores={
            m.name: m(y[onestep_predicted], onestep[onestep_predicted]) for m in metrics
        },
    )


def _horizon_scores(
    y: np.ndarray, pred: np.ndarray, steps: np.ndarray, metric_names: list[str]
) -> pl.DataFrame:
    """予測ステップごとに全foldを通した指標を計算する。"""
    rows = []
    for h in np.unique(steps[steps > 0]):
        mask = steps == h
        row: dict[str, Any] = {"step": int(h), "n": int(mask.sum())}
        for name in metric_names:
            try:
                row[name] = get_metric(name)(y[mask], pred[mask])
            except ValueError:
                # 行数が少なく計算できない指標（r2の1行など）は欠損にする
                row[name] = None
        rows.append(row)
    return pl.DataFrame(rows)
