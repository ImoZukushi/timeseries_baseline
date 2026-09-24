"""予測タスクの種類と、タスクに応じた目的変数の前処理・予測値の取り出し。"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import numpy as np


class Task(StrEnum):
    """予測タスクの種類。

    `TIME_SERIES` は「ラグ等で特徴量化した上での回帰」として扱う。
    学習・予測の仕組みは `REGRESSION` と同じで、CV分割に時間順序を強制する点だけが異なる。
    """

    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"


def is_classification(task: Task) -> bool:
    """分類タスクかどうかを返す。"""
    return task in (Task.BINARY, Task.MULTICLASS)


def encode_target(y: np.ndarray, task: Task) -> tuple[np.ndarray, np.ndarray | None]:
    """目的変数をモデルに渡せる形式に変換する。

    分類タスクではラベルを 0..K-1 の整数に変換する（XGBoost等が整数ラベルを要求するため）。
    回帰タスクではfloatに変換するのみ。

    Args:
        y: 目的変数の1次元配列。
        task: 予測タスク。

    Returns:
        (変換後の目的変数, 元のクラスラベル配列)。回帰ではクラスラベルはNone。

    Raises:
        ValueError: 目的変数に欠損がある場合、または二値分類でクラス数が2でない場合。
    """
    if is_classification(task):
        if any(v is None for v in y.tolist()):
            raise ValueError("目的変数に欠損値が含まれています")
        classes, encoded = np.unique(y, return_inverse=True)
        if task is Task.BINARY and len(classes) != 2:
            raise ValueError(f"二値分類のクラス数が2ではありません: {len(classes)}")
        return encoded.astype(np.int64), classes
    y_float = y.astype(np.float64)
    if np.isnan(y_float).any():
        raise ValueError("目的変数に欠損値が含まれています")
    return y_float, None


def predict(estimator: Any, X: Any, task: Task, n_classes: int | None = None) -> np.ndarray:
    """タスクに応じた予測値を返す。

    - 二値分類: 陽性クラス（ラベル1）の確率 (n_samples,)
    - 多クラス分類: 各クラスの確率 (n_samples, n_classes)
    - 回帰・時系列: 予測値 (n_samples,)

    多クラス分類では、学習foldに一部のクラスが含まれない場合に `predict_proba` の列数が
    全クラス数より少なくなるため、`estimator.classes_` を使って全クラスの列に並べ直す。

    Args:
        estimator: fit済みのsklearn互換estimator（またはPipeline）。
        X: 予測対象の特徴量。
        task: 予測タスク。
        n_classes: 多クラス分類の全クラス数。

    Returns:
        予測値の配列。
    """
    if task is Task.BINARY:
        proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)
        classes = list(estimator.classes_)
        if 1 not in classes:
            # 学習foldに陽性クラスが無い場合は陽性確率0とする
            return np.zeros(proba.shape[0])
        return proba[:, classes.index(1)]
    if task is Task.MULTICLASS:
        if n_classes is None:
            raise ValueError("多クラス分類では n_classes の指定が必要です")
        proba = np.asarray(estimator.predict_proba(X), dtype=np.float64)
        full = np.zeros((proba.shape[0], n_classes))
        full[:, np.asarray(estimator.classes_, dtype=np.int64)] = proba
        # float32で確率を返すモデル（XGBoost）の丸め誤差で行和が1からずれるため正規化する
        row_sum = full.sum(axis=1, keepdims=True)
        return np.divide(full, row_sum, out=full, where=row_sum > 0)
    return np.asarray(estimator.predict(X), dtype=np.float64)
