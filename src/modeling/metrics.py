"""評価指標のレジストリ。

指標は `(y_true, y_pred) -> float` の関数として定義する。`y_pred` は
`modeling.tasks.predict` の出力形式（二値分類は陽性確率、多クラスは確率行列、回帰は予測値）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn import metrics as skm


@dataclass(frozen=True)
class Metric:
    """評価指標の定義。

    Attributes:
        name: 指標名（YAMLで指定する名前）。
        func: `(y_true, y_pred) -> float` の関数。
        greater_is_better: 大きいほど良い指標ならTrue。
    """

    name: str
    func: Callable[[np.ndarray, np.ndarray], float]
    greater_is_better: bool

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """指標を計算する。"""
        return float(self.func(y_true, y_pred))

    @property
    def direction(self) -> Literal["maximize", "minimize"]:
        """Optunaの `direction` 引数に渡す文字列（"maximize" / "minimize"）。"""
        return "maximize" if self.greater_is_better else "minimize"


def _to_label(y_pred: np.ndarray) -> np.ndarray:
    """確率予測をクラスラベルに変換する（二値は閾値0.5、多クラスはargmax）。"""
    if y_pred.ndim == 2:
        return np.argmax(y_pred, axis=1)
    return (y_pred >= 0.5).astype(np.int64)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))


def _auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_pred.ndim == 2:
        # 多クラスはOne-vs-Restのマクロ平均
        return float(
            skm.roc_auc_score(y_true, y_pred, multi_class="ovr", labels=range(y_pred.shape[1]))
        )
    return float(skm.roc_auc_score(y_true, y_pred))


def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = list(range(y_pred.shape[1])) if y_pred.ndim == 2 else [0, 1]
    return float(skm.log_loss(y_true, y_pred, labels=labels))


_METRICS: dict[str, Metric] = {
    m.name: m
    for m in (
        # 回帰
        Metric("rmse", _rmse, greater_is_better=False),
        Metric("mse", lambda t, p: skm.mean_squared_error(t, p), greater_is_better=False),
        Metric("mae", lambda t, p: skm.mean_absolute_error(t, p), greater_is_better=False),
        Metric("r2", lambda t, p: skm.r2_score(t, p), greater_is_better=True),
        # 分類
        Metric("auc", _auc, greater_is_better=True),
        Metric("logloss", _logloss, greater_is_better=False),
        Metric(
            "accuracy", lambda t, p: skm.accuracy_score(t, _to_label(p)), greater_is_better=True
        ),
        Metric("f1", lambda t, p: skm.f1_score(t, _to_label(p)), greater_is_better=True),
        Metric(
            "f1_macro",
            lambda t, p: skm.f1_score(t, _to_label(p), average="macro"),
            greater_is_better=True,
        ),
    )
}


def get_metric(name: str) -> Metric:
    """名前から評価指標を取得する。

    Raises:
        KeyError: 未登録の指標名の場合。
    """
    if name not in _METRICS:
        raise KeyError(f"未登録の評価指標です: {name}（利用可能: {sorted(_METRICS)}）")
    return _METRICS[name]


def available_metrics() -> list[str]:
    """登録済みの評価指標名の一覧を返す。"""
    return sorted(_METRICS)
