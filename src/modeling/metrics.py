"""評価指標のレジストリ。

指標は `(y_true, y_pred) -> float` の関数として定義する。`y_pred` は
`modeling.tasks.predict` の出力形式（二値分類は陽性確率、多クラスは確率行列、回帰は予測値）。
クラスラベルが必要な指標（accuracy・precision等）は、二値は閾値0.5、多クラスはargmaxで
確率をラベルに変換してから計算する。

名前の書式:
    - 通常: `rmse`, `pr_auc`, `f1_macro` など（`available_metrics()` で一覧）
    - 別名: `auc` は `roc_auc` と同じ指標
    - パラメータ付き: `pauc@0.05`（FPRの上限を0.05にした部分AUC。`pauc` の既定は0.1）
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Literal

import numpy as np
from sklearn import metrics as skm
from sklearn.preprocessing import label_binarize

from modeling.tasks import Task

_REGRESSION = frozenset({Task.REGRESSION, Task.TIME_SERIES})
_CLASSIFICATION = frozenset({Task.BINARY, Task.MULTICLASS})
_BINARY = frozenset({Task.BINARY})

# pAUCのFPR上限の既定値
DEFAULT_PAUC_MAX_FPR = 0.1


@dataclass(frozen=True)
class Metric:
    """評価指標の定義。

    Attributes:
        name: 指標名（YAMLで指定する名前）。
        func: `(y_true, y_pred) -> float` の関数。
        greater_is_better: 大きいほど良い指標ならTrue。
        tasks: この指標を使えるタスク。
    """

    name: str
    func: Callable[[np.ndarray, np.ndarray], float]
    greater_is_better: bool
    tasks: frozenset[Task]

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """指標を計算する。"""
        return float(self.func(y_true, y_pred))

    @property
    def direction(self) -> Literal["maximize", "minimize"]:
        """Optunaの `direction` 引数に渡す文字列（"maximize" / "minimize"）。"""
        return "maximize" if self.greater_is_better else "minimize"


# --- 共通ヘルパー --------------------------------------------------------------------


def _to_label(y_pred: np.ndarray) -> np.ndarray:
    """確率予測をクラスラベルに変換する（二値は閾値0.5、多クラスはargmax）。"""
    if y_pred.ndim == 2:
        return np.argmax(y_pred, axis=1)
    return (y_pred >= 0.5).astype(np.int64)


def _labels(y_pred: np.ndarray) -> list[int]:
    """全クラスのラベル（検証foldに一部のクラスが無くても全クラスを対象にするため）。"""
    return list(range(y_pred.shape[1])) if y_pred.ndim == 2 else [0, 1]


# --- 回帰 -------------------------------------------------------------------------------


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(skm.mean_squared_error(y_true, y_pred)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """平均絶対パーセント誤差（比率。0.1 = 10%）。

    目的変数が0の行では分母が0になり、sklearnの実装では極端に大きな値になる。
    0を多く含む目的変数には向かない。
    """
    return float(skm.mean_absolute_percentage_error(y_true, y_pred))


def _rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """対数二乗平均平方根誤差。負の予測は0に切り詰める。

    Raises:
        ValueError: 目的変数に負の値がある場合（対数が定義できないため）。
    """
    if (y_true < 0).any():
        raise ValueError("RMSLEは負の目的変数には使えません")
    # 回帰モデルはわずかに負の値を出すことがあるため、CVの途中で落ちないよう0に切り詰める
    return float(np.sqrt(skm.mean_squared_log_error(y_true, np.clip(y_pred, 0, None))))


# --- 分類（ラベル） ------------------------------------------------------------------


def _averaged(
    score_fn: Callable[..., float], average: str, y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """precision / recall / F1 を指定の平均方法で計算する（予測0件のクラスは0とする）。"""
    return float(
        score_fn(
            y_true, _to_label(y_pred), labels=_labels(y_pred), average=average, zero_division=0
        )
    )


def _binary_score(score_fn: Callable[..., float], y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """陽性クラス（ラベル1）の precision / recall / F1。"""
    return float(score_fn(y_true, _to_label(y_pred), zero_division=0))


def _mcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(skm.matthews_corrcoef(y_true, _to_label(y_pred)))


def _g_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """各クラスのRecallの幾何平均（二値では sqrt(TPR × TNR)）。

    検証データに存在しないクラスのRecallは0として扱う（その場合G-Meanは0になる）。
    """
    recalls = skm.recall_score(
        y_true, _to_label(y_pred), labels=_labels(y_pred), average=None, zero_division=0
    )
    return float(np.prod(recalls) ** (1.0 / len(recalls)))


# --- 分類（確率） ------------------------------------------------------------------


def _roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_pred.ndim == 2:
        # 多クラスはOne-vs-Restのマクロ平均
        return float(skm.roc_auc_score(y_true, y_pred, multi_class="ovr", labels=_labels(y_pred)))
    return float(skm.roc_auc_score(y_true, y_pred))


def _roc_auc_averaged(average: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """micro / macro 平均のROC-AUC（二値では通常のROC-AUCと同じ）。

    多クラスではラベルをone-hot化し、各クラスを「そのクラス vs それ以外」とみなして計算する。
    macroでは、検証データに存在しないクラスはAUCが定義できないため平均から除外する。
    """
    if y_pred.ndim == 1:
        return float(skm.roc_auc_score(y_true, y_pred))
    onehot = label_binarize(y_true, classes=_labels(y_pred))
    if average == "macro":
        present = onehot.sum(axis=0) > 0
        onehot, y_pred = onehot[:, present], y_pred[:, present]
    return float(skm.roc_auc_score(onehot, y_pred, average=average))


def _pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PR曲線下面積（平均適合率 average precision で近似）。"""
    return float(skm.average_precision_score(y_true, y_pred))


def _pauc(max_fpr: float, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FPRが `max_fpr` 以下の範囲の部分AUC（McClish補正で標準化。0.5=ランダム、1=完全）。"""
    return float(skm.roc_auc_score(y_true, y_pred, max_fpr=max_fpr))


def _logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(skm.log_loss(y_true, y_pred, labels=_labels(y_pred)))


# --- レジストリ ------------------------------------------------------------------------


def _build_registry() -> dict[str, Metric]:
    metrics = [
        # 回帰
        Metric("rmse", _rmse, False, _REGRESSION),
        Metric("mse", lambda t, p: skm.mean_squared_error(t, p), False, _REGRESSION),
        Metric("mae", lambda t, p: skm.mean_absolute_error(t, p), False, _REGRESSION),
        Metric("mape", _mape, False, _REGRESSION),
        Metric("rmsle", _rmsle, False, _REGRESSION),
        Metric("r2", lambda t, p: skm.r2_score(t, p), True, _REGRESSION),
        # 分類（ラベル）
        Metric("accuracy", lambda t, p: skm.accuracy_score(t, _to_label(p)), True, _CLASSIFICATION),
        Metric("mcc", _mcc, True, _CLASSIFICATION),
        Metric("g_mean", _g_mean, True, _CLASSIFICATION),
        Metric("precision", partial(_binary_score, skm.precision_score), True, _BINARY),
        Metric("recall", partial(_binary_score, skm.recall_score), True, _BINARY),
        Metric("f1", partial(_binary_score, skm.f1_score), True, _BINARY),
        # 分類（確率）
        Metric("roc_auc", _roc_auc, True, _CLASSIFICATION),
        Metric("roc_auc_micro", partial(_roc_auc_averaged, "micro"), True, _CLASSIFICATION),
        Metric("roc_auc_macro", partial(_roc_auc_averaged, "macro"), True, _CLASSIFICATION),
        Metric("pr_auc", _pr_auc, True, _BINARY),
        Metric("pauc", partial(_pauc, DEFAULT_PAUC_MAX_FPR), True, _BINARY),
        Metric("logloss", _logloss, False, _CLASSIFICATION),
    ]
    # precision / recall / F1 の micro・macro・weighted 平均
    for base, score_fn in (
        ("precision", skm.precision_score),
        ("recall", skm.recall_score),
        ("f1", skm.f1_score),
    ):
        for average in ("micro", "macro", "weighted"):
            metrics.append(
                Metric(
                    f"{base}_{average}",
                    partial(_averaged, score_fn, average),
                    True,
                    _CLASSIFICATION,
                )
            )
    return {m.name: m for m in metrics}


_METRICS = _build_registry()

# 別名 → 正式名
_ALIASES = {"auc": "roc_auc"}


def _parse_pauc(name: str, param: str) -> Metric:
    """`pauc@<max_fpr>` から部分AUCの指標を作る。

    Raises:
        ValueError: `max_fpr` が数値でない、または (0, 1] の範囲外の場合。
    """
    try:
        max_fpr = float(param)
    except ValueError as e:
        raise ValueError(f"{name}: FPRの上限は数値で指定してください（例: pauc@0.05）") from e
    if not 0 < max_fpr <= 1:
        raise ValueError(f"{name}: FPRの上限は 0 < max_fpr <= 1 で指定してください")
    return Metric(name, partial(_pauc, max_fpr), True, _BINARY)


def get_metric(name: str) -> Metric:
    """名前から評価指標を取得する。

    返す `Metric.name` は指定した名前のまま（別名・パラメータ付きの名前でも）とする。
    CV結果の指標名が設定ファイルの記述と一致するようにするため。

    Raises:
        KeyError: 未登録の指標名の場合。
        ValueError: パラメータ付きの指標でパラメータが不正な場合。
    """
    base, sep, param = name.partition("@")
    if sep:
        if base != "pauc":
            raise KeyError(f"パラメータを指定できるのは pauc のみです: {name}")
        return _parse_pauc(name, param)
    canonical = _ALIASES.get(name, name)
    if canonical not in _METRICS:
        raise KeyError(f"未登録の評価指標です: {name}（利用可能: {available_metrics()}）")
    return dataclasses.replace(_METRICS[canonical], name=name)


def available_metrics() -> list[str]:
    """登録済みの評価指標名の一覧を返す（別名を含む。`pauc@<max_fpr>` は `pauc` として表示）。"""
    return sorted([*_METRICS, *_ALIASES])
