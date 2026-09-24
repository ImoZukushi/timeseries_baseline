"""学習用データ一式（`Dataset`）と、それに対するCV学習の実行。

`experiment`（1実験の実行）と `tuning`（チューニング）の両方から使うため、
循環importを避けて独立したモジュールにしている。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from modeling.config import ExperimentConfig
from modeling.cv import Fold
from modeling.forecasting import recursive_backtest
from modeling.trainer import CVResult, FoldCallback, run_cv


@dataclass
class Dataset:
    """学習・予測に使うデータ一式。

    Attributes:
        X: 学習データの特徴量（`forecast` 指定時は目的変数の特徴量も含む）。
        y: エンコード済みの目的変数。
        classes: 分類タスクの元のクラスラベル（回帰ではNone）。
        folds: CV分割。
        ids: 学習データの行ID（`data.id_col` 指定時）。
        X_test: テストデータの特徴量（`forecast` 指定時は再帰的に作るためNone）。
        ids_test: テストデータの行ID。
        frame: 学習データ全体（`forecast` 指定時のみ。目的変数・時刻・系列列を含む）。
        test_frame: テストデータ全体（`forecast` 指定時のみ）。
    """

    X: pl.DataFrame
    y: np.ndarray
    classes: np.ndarray | None
    folds: list[Fold]
    ids: pl.Series | None = None
    X_test: pl.DataFrame | None = None
    ids_test: pl.Series | None = None
    frame: pl.DataFrame | None = None
    test_frame: pl.DataFrame | None = None

    @property
    def n_classes(self) -> int | None:
        """分類タスクのクラス数。"""
        return None if self.classes is None else len(self.classes)


def cross_validate(
    config: ExperimentConfig,
    dataset: Dataset,
    params: dict[str, Any] | None = None,
    fold_callback: FoldCallback | None = None,
    predict_test: bool = True,
) -> CVResult:
    """設定に応じてCV学習を行う（`forecast` 指定時は再帰バックテスト、それ以外は `run_cv`）。

    Args:
        config: 実験設定。
        dataset: 学習用データ一式。
        params: モデルパラメータ（Noneなら設定値）。
        fold_callback: fold完了ごとに `(fold番号, 主指標スコア)` で呼ばれる関数。
        predict_test: テストデータも予測するか（チューニング中は不要なのでFalse）。

    Returns:
        CV結果（`forecast` 指定時は `ForecastCVResult`）。
    """
    if config.forecast is not None:
        if dataset.frame is None:
            raise ValueError("forecast 指定時は Dataset.frame が必要です（prepare_dataset を使う）")
        return recursive_backtest(
            config,
            dataset.frame,
            dataset.X,
            dataset.y,
            dataset.folds,
            test_frame=dataset.test_frame if predict_test else None,
            params=params,
            fold_callback=fold_callback,
        )
    return run_cv(
        config,
        dataset.X,
        dataset.y,
        dataset.folds,
        X_test=dataset.X_test if predict_test else None,
        params=params,
        n_classes=dataset.n_classes,
        fold_callback=fold_callback,
    )
