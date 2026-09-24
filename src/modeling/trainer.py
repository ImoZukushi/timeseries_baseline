"""CV学習ループ。

fold毎に新しいPipelineを作り、**学習foldのみで** 特徴量エンジニアリングとモデルを
fitする（検証fold・テストデータは `transform` / `predict` のみ）。これにより
ターゲットエンコーディング等の前処理を含めてリークが構造的に起きない。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline

from modeling.config import ExperimentConfig
from modeling.cv import Fold, folds_to_ids
from modeling.metrics import get_metric
from modeling.models import get_model_spec
from modeling.pipeline import MODEL_STEP, build_pipeline
from modeling.tasks import Task, predict

# fold完了ごとに呼ばれるコールバック（fold番号, そのfoldの主指標スコア）。
# Optunaの途中打ち切り（pruning）に使う。
FoldCallback = Callable[[int, float], None]


@dataclass
class CVResult:
    """CV学習の結果。

    Attributes:
        oof_pred: OOF予測。二値・回帰は (n_samples,)、多クラスは (n_samples, n_classes)。
            どの検証foldにも入らない行（時系列CVの最初の期間等）はNaN。
        fold_ids: 各行が検証データになったfold番号（どのfoldにも入らない行は-1）。
        fold_scores: 指標名 → fold別スコアのリスト。
        oof_scores: 指標名 → 予測された全行を通したOOFスコア。
        test_pred: テストデータの予測（テストデータが無ければNone）。
        models: fold毎の学習済みPipeline（`refit_full` なら全データで学習したものが末尾に追加）。
        best_iterations: fold毎のearly stopping最良イテレーション数（該当しなければNone）。
        params: 学習に使ったモデルパラメータ。
    """

    oof_pred: np.ndarray
    fold_ids: np.ndarray
    fold_scores: dict[str, list[float]]
    oof_scores: dict[str, float]
    test_pred: np.ndarray | None
    models: list[Pipeline]
    best_iterations: list[int | None] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def mean_scores(self) -> dict[str, float]:
        """指標ごとのfold平均スコア。"""
        return {k: float(np.mean(v)) for k, v in self.fold_scores.items()}

    def std_scores(self) -> dict[str, float]:
        """指標ごとのfold間標準偏差。"""
        return {k: float(np.std(v)) for k, v in self.fold_scores.items()}


def fit_pipeline(
    config: ExperimentConfig,
    X_train: pl.DataFrame,
    y_train: np.ndarray,
    X_valid: pl.DataFrame | None = None,
    y_valid: np.ndarray | None = None,
    params: dict[str, Any] | None = None,
    early_stopping: bool = True,
) -> Pipeline:
    """Pipelineを1つ学習する。

    early stoppingのために検証データをモデルに渡す場合、検証データも学習データで
    fitした前処理で変換してから渡す必要があるため、前処理とモデルを分けてfitする。

    Args:
        config: 実験設定。
        X_train: 学習データの特徴量。
        y_train: 学習データの目的変数。
        X_valid: early stopping用の検証データ（Noneならearly stoppingしない）。
        y_valid: early stopping用の検証データの目的変数。
        params: モデルパラメータ（Noneなら設定値）。
        early_stopping: Falseならearly stoppingを無効化する（全データ再学習時）。

    Returns:
        学習済みPipeline。
    """
    spec = get_model_spec(config.model.name)
    rounds = config.model.early_stopping_rounds if early_stopping else None
    if X_valid is None:
        rounds = None
    run_config = config.model_copy(
        update={"model": config.model.model_copy(update={"early_stopping_rounds": rounds})}
    )
    pipeline = build_pipeline(run_config, params)
    preprocess = pipeline[:-1]
    Xt_train = preprocess.fit_transform(X_train, y_train)
    fit_kwargs: dict[str, Any] = {}
    if rounds is not None and X_valid is not None:
        Xt_valid = preprocess.transform(X_valid)
        fit_kwargs = spec.fit_kwargs(Xt_valid, y_valid, rounds)
    pipeline.named_steps[MODEL_STEP].fit(Xt_train, y_train, **fit_kwargs)
    return pipeline


def run_cv(
    config: ExperimentConfig,
    X: pl.DataFrame,
    y: np.ndarray,
    folds: Sequence[Fold],
    X_test: pl.DataFrame | None = None,
    params: dict[str, Any] | None = None,
    n_classes: int | None = None,
    fold_callback: FoldCallback | None = None,
) -> CVResult:
    """CVで学習し、OOF予測・fold別スコア・テスト予測を返す。

    Args:
        config: 実験設定。
        X: 学習データの特徴量（polars DataFrame）。
        y: エンコード済みの目的変数（`modeling.tasks.encode_target` の出力）。
        folds: `(train_idx, valid_idx)` のリスト（`modeling.cv.make_folds` の出力）。
        X_test: テストデータの特徴量（任意）。
        params: モデルパラメータ（Noneなら設定値）。
        n_classes: 多クラス分類のクラス数。
        fold_callback: fold完了ごとに `(fold番号, 主指標スコア)` で呼ばれる関数。

    Returns:
        CV結果。
    """
    task = config.task
    if task is Task.MULTICLASS and n_classes is None:
        n_classes = int(np.max(y)) + 1
    n = X.height
    oof_shape: tuple[int, ...] = (
        (n,) if n_classes is None or task is not Task.MULTICLASS else (n, n_classes)
    )
    oof_pred = np.full(oof_shape, np.nan)
    metrics = [get_metric(name) for name in config.metrics]
    fold_scores: dict[str, list[float]] = {m.name: [] for m in metrics}
    models: list[Pipeline] = []
    best_iterations: list[int | None] = []
    test_preds: list[np.ndarray] = []
    spec = get_model_spec(config.model.name)
    use_params = config.model.params if params is None else params

    for i, (train_idx, valid_idx) in enumerate(folds):
        X_tr, X_va = X[train_idx], X[valid_idx]
        y_tr, y_va = y[train_idx], y[valid_idx]
        pipeline = fit_pipeline(config, X_tr, y_tr, X_va, y_va, params=use_params)
        pred = predict(pipeline, X_va, task, n_classes)
        oof_pred[valid_idx] = pred
        for m in metrics:
            fold_scores[m.name].append(m(y_va, pred))
        models.append(pipeline)
        best_iterations.append(spec.best_iteration(pipeline.named_steps[MODEL_STEP]))
        if X_test is not None and config.test_prediction == "fold_mean":
            test_preds.append(predict(pipeline, X_test, task, n_classes))
        if fold_callback is not None:
            fold_callback(i, fold_scores[config.primary_metric][-1])

    fold_ids = folds_to_ids(folds, n)
    predicted = fold_ids >= 0
    oof_scores = {m.name: m(y[predicted], oof_pred[predicted]) for m in metrics}

    test_pred: np.ndarray | None = None
    if X_test is not None:
        if config.test_prediction == "fold_mean":
            test_pred = np.mean(test_preds, axis=0)
        else:
            full_params = use_params
            iters = [b for b in best_iterations if b is not None]
            if iters:
                # early stoppingを使った場合は、fold平均の最良イテレーション数で全データを学習
                full_params = spec.with_n_iterations(use_params, int(np.mean(iters)))
            full = fit_pipeline(config, X, y, params=full_params, early_stopping=False)
            models.append(full)
            test_pred = predict(full, X_test, task, n_classes)

    return CVResult(
        oof_pred=oof_pred,
        fold_ids=fold_ids,
        fold_scores=fold_scores,
        oof_scores=oof_scores,
        test_pred=test_pred,
        models=models,
        best_iterations=best_iterations,
        params=dict(use_params),
    )
