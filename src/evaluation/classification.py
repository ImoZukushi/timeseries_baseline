"""分類モデルの評価の可視化（混同行列・ROC曲線・PR曲線）。

scikit-learn の `ConfusionMatrixDisplay` / `RocCurveDisplay` / `PrecisionRecallDisplay` を
内部で使い、日本語の見出しと多クラス分類（One-vs-Rest: 各クラス vs それ以外）への対応を加える。

入力の形:
    - `y_true`: クラス番号（0..K-1。`modeling.tasks.encode_target` の出力と同じ）
    - `y_score`: 二値は陽性クラスの確率 (n,)、多クラスは各クラスの確率 (n, K)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
from sklearn import metrics as skm

from evaluation._common import class_labels, new_axes
from util.plotting import ensure_japanese_font


def scores_to_labels(y_score: Any, threshold: float = 0.5) -> np.ndarray:
    """確率をクラス番号にする（二値は `threshold` 以上を1、多クラスは確率が最大のクラス）。"""
    score = np.asarray(y_score, dtype=np.float64)
    if score.ndim == 2:
        return np.argmax(score, axis=1)
    return (score >= threshold).astype(np.int64)


def _n_classes(y_score: np.ndarray) -> int:
    return y_score.shape[1] if y_score.ndim == 2 else 2


def _valid_rows(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """予測が欠損（NaN）の行を除く。"""
    flat = y_score if y_score.ndim == 1 else y_score.sum(axis=1)
    return np.isfinite(flat) & np.isfinite(y_true.astype(np.float64))


class ConfusionMatrixDisplay:
    """混同行列（件数 ／ 実測クラスごとの割合＝再現率）の2枚組。

    描画には sklearn の `ConfusionMatrixDisplay` を使う。

    Attributes:
        matrix: 件数の混同行列（行: 実測, 列: 予測）。
        labels: クラスの表示名。
        threshold: 二値分類で陽性とみなす確率の閾値。
        figure_: 描画したFigure。
        ax_: 描画したAxes（2個）。
    """

    def __init__(self, matrix: np.ndarray, labels: list[str], threshold: float | None) -> None:
        self.matrix = matrix
        self.labels = labels
        self.threshold = threshold

    @classmethod
    def from_predictions(
        cls,
        y_true: Any,
        y_score: Any,
        *,
        class_names: Sequence[Any] | None = None,
        threshold: float = 0.5,
        ax: Any = None,
    ) -> Self:
        """実測クラスと予測確率（またはクラス番号）から描く。"""
        t = np.asarray(y_true)
        score = np.asarray(y_score, dtype=np.float64)
        mask = _valid_rows(t, score)
        k = _n_classes(score)
        predicted = scores_to_labels(score[mask], threshold)
        matrix = skm.confusion_matrix(t[mask].astype(np.int64), predicted, labels=list(range(k)))
        return cls(
            matrix, class_labels(k, class_names), threshold if score.ndim == 1 else None
        ).plot(ax=ax)

    def plot(self, ax: Any = None) -> Self:
        """混同行列を描く（`ax` を渡す場合は2個の並び）。"""
        ensure_japanese_font()
        k = len(self.labels)
        fig, axes = new_axes(ax, ncols=2, figsize=(6 + 1.2 * k, 3 + 0.6 * k))
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = self.matrix / self.matrix.sum(axis=1, keepdims=True)
        normalized = np.nan_to_num(normalized)
        for a, values, fmt, title in (
            (axes.flat[0], self.matrix, "d", "件数"),
            (axes.flat[1], normalized, ".2f", "実測クラスごとの割合（対角＝再現率）"),
        ):
            skm.ConfusionMatrixDisplay(values, display_labels=self.labels).plot(
                ax=a, values_format=fmt, colorbar=False, cmap="Blues"
            )
            a.set_xlabel("予測クラス")
            a.set_ylabel("実測クラス")
            a.set_title(title)
        accuracy = np.trace(self.matrix) / max(self.matrix.sum(), 1)
        rule = (
            f"確率 {self.threshold} 以上を陽性"
            if self.threshold is not None
            else "確率が最大のクラス"
        )
        fig.suptitle(f"混同行列（予測: {rule}, 正解率={accuracy:.3f}, n={self.matrix.sum():,}）")
        self.figure_, self.ax_ = fig, axes
        return self


class _CurveDisplay:
    """ROC曲線・PR曲線の共通部分（二値は1本、多クラスはクラスごと）。"""

    _sklearn_display: Any = None
    _score_name = ""
    _title = ""
    _xlabel = ""
    _ylabel = ""
    # 多クラスでも1本だけ基準線を描くか（ROCの対角線はクラスによらず同じ。PRはクラスごとに違う）
    _shared_chance_level = False
    figure_: Any
    ax_: Any

    def __init__(self, scores: dict[str, float], macro: float, labels: list[str]) -> None:
        self.scores = scores
        self.macro = macro
        self.labels = labels

    @classmethod
    def _score(cls, y_true: np.ndarray, y_score: np.ndarray) -> float:
        raise NotImplementedError

    @classmethod
    def from_predictions(
        cls,
        y_true: Any,
        y_score: Any,
        *,
        class_names: Sequence[Any] | None = None,
        ax: Any = None,
    ) -> Self:
        """実測クラスと予測確率から描く。"""
        ensure_japanese_font()
        t = np.asarray(y_true)
        score = np.asarray(y_score, dtype=np.float64)
        mask = _valid_rows(t, score)
        t, score = t[mask].astype(np.int64), score[mask]
        k = _n_classes(score)
        labels = class_labels(k, class_names)
        fig, ax = new_axes(ax, figsize=(7, 6))
        scores: dict[str, float] = {}
        if score.ndim == 1:
            # 二値分類: 陽性クラス（ラベル1）の曲線を1本
            cls._sklearn_display.from_predictions(
                t, score, pos_label=1, name=f"陽性={labels[1]}", ax=ax, plot_chance_level=True
            )
            scores[labels[1]] = cls._score(t, score)
        else:
            # 多クラス分類: One-vs-Rest（各クラス vs それ以外）でクラスごとに描く
            for c in range(k):
                binary = (t == c).astype(np.int64)
                if binary.min() == binary.max():
                    # 検証データにそのクラスが無い（または全部そのクラス）場合は曲線を引けない
                    continue
                cls._sklearn_display.from_predictions(
                    binary,
                    score[:, c],
                    pos_label=1,
                    name=f"{labels[c]} vs 他",
                    ax=ax,
                    plot_chance_level=cls._shared_chance_level and not scores,
                )
                scores[labels[c]] = cls._score(binary, score[:, c])
        macro = float(np.mean(list(scores.values()))) if scores else float("nan")
        disp = cls(scores, macro, labels)
        ax.set_title(f"{cls._title}（{cls._score_name} macro平均={macro:.3f}, n={t.size:,}）")
        ax.set_xlabel(cls._xlabel)
        ax.set_ylabel(cls._ylabel)
        ax.legend(loc="lower right" if cls is RocCurveDisplay else "lower left", fontsize=9)
        disp.figure_, disp.ax_ = fig, ax
        return disp


class RocCurveDisplay(_CurveDisplay):
    """ROC曲線（偽陽性率 vs 真陽性率）。凡例にAUC、点線はランダムな予測。

    Attributes:
        scores: クラスの表示名 → ROC-AUC。
        macro: クラス平均のROC-AUC。
        figure_: 描画したFigure。
        ax_: 描画したAxes。
    """

    _sklearn_display = skm.RocCurveDisplay
    _score_name = "AUC"
    _title = "ROC曲線"
    _xlabel = "偽陽性率（陰性を誤って陽性とした割合）"
    _ylabel = "真陽性率（再現率）"
    _shared_chance_level = True

    @classmethod
    def _score(cls, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return float(skm.roc_auc_score(y_true, y_score))


class PrecisionRecallDisplay(_CurveDisplay):
    """PR曲線（再現率 vs 適合率）。凡例にAP（平均適合率）、点線は陽性率（ランダムな予測の水準）。

    陽性が少ない（不均衡な）データでは、ROC曲線より性能の差が見えやすい。

    Attributes:
        scores: クラスの表示名 → AP。
        macro: クラス平均のAP。
        figure_: 描画したFigure。
        ax_: 描画したAxes。
    """

    _sklearn_display = skm.PrecisionRecallDisplay
    _score_name = "AP"
    _title = "PR曲線"
    _xlabel = "再現率（陽性のうち予測で拾えた割合）"
    _ylabel = "適合率（陽性と予測したうち当たった割合）"

    @classmethod
    def _score(cls, y_true: np.ndarray, y_score: np.ndarray) -> float:
        return float(skm.average_precision_score(y_true, y_score))
