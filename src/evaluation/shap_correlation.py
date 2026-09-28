"""SHAP値の相関の可視化。

2種類の「相関」を扱う。

1. **特徴量の値と SHAP 値の相関**（`shap_feature_correlation` / `ShapCorrelationBarDisplay` /
   `ShapDependenceDisplay`）: 各特徴量について「値が大きいほど予測を押し上げるか・押し下げるか」。
   平均|SHAP|の重要度だけでは効きの向きが分からないため、相関の符号で重要度を色分けし、
   値 vs SHAP値 の散布図（dependence plot）で関係の形（直線的か、しきい値があるか等）を見る。
2. **特徴量同士の SHAP 値の相関**（`shap_value_correlation` / `ShapValueCorrelationDisplay` /
   `ShapScatterMatrixDisplay`）: 2つの特徴量の SHAP 値が同じ向きに動く（似た情報を持つ・冗長）か、
   逆向きに動く（打ち消し合う）かを、相関行列のヒートマップと SHAP 値同士の散布図で見る。

入力はいずれも、SHAP値 (n, 特徴量数)・特徴量の値 (n, 特徴量数)・特徴量名。多クラス分類では
クラスごとの SHAP 値（(n, 特徴量数) に切り出したもの）を渡す。欠損を含む特徴量でも、
有効な組だけで相関を計算する（計算できない場合は欠損）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
from scipy import stats

from evaluation._common import new_axes, subsample_index
from util.plotting import ensure_japanese_font

# 相関の向きの色（値が大きいほど予測を上げる / 下げる / 相関が計算できない）
_POSITIVE, _NEGATIVE, _UNKNOWN = "#c44e52", "#4c72b0", "#b0b0b0"


def _as_2d(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[:, None] if arr.ndim == 1 else arr


def _pair_correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """有効な組だけで Pearson・Spearman の相関係数を計算する（計算できなければ NaN）。"""
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        # 点が少ない、またはどちらかが一定の場合は相関が定義できない
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(a, b)[0, 1])
    spearman = float(stats.spearmanr(a, b).statistic)
    return pearson, spearman


def _direction(r: float, threshold: float = 0.05) -> str:
    if not np.isfinite(r):
        return "なし"
    if r > threshold:
        return "正"
    if r < -threshold:
        return "負"
    return "なし"


def shap_feature_correlation(
    shap_values: Any, data: Any, feature_names: Sequence[str]
) -> pl.DataFrame:
    """特徴量ごとに、平均|SHAP|と「特徴量の値と SHAP 値の相関」をまとめた表を返す。

    Args:
        shap_values: SHAP値 (n, 特徴量数)。
        data: 特徴量の値 (n, 特徴量数)。モデルに入力した値（前処理後）。
        feature_names: 特徴量名。

    Returns:
        列 `feature`, `mean_abs_shap`, `pearson`, `spearman`, `direction`（正/負/なし）の表
        （平均|SHAP|の降順）。`direction` は Spearman（順位相関）の符号で決める
        （単調だが直線的でない関係も拾うため）。
    """
    values, x = _as_2d(shap_values), _as_2d(data)
    rows = []
    for j, name in enumerate(feature_names):
        pearson, spearman = _pair_correlation(x[:, j], values[:, j])
        rows.append(
            {
                "feature": str(name),
                "mean_abs_shap": float(np.nanmean(np.abs(values[:, j]))),
                "pearson": pearson,
                "spearman": spearman,
                "direction": _direction(spearman),
            }
        )
    return (
        pl.DataFrame(rows)
        .with_columns(pl.selectors.float().fill_nan(None))
        .sort("mean_abs_shap", descending=True)
    )


def shap_value_correlation(shap_values: Any, feature_names: Sequence[str]) -> pl.DataFrame:
    """特徴量同士の SHAP 値の相関行列（Pearson）を返す。

    Returns:
        先頭列 `feature` と、各特徴量名の列からなる正方行列の表（計算できない組は欠損）。
    """
    values = _as_2d(shap_values)
    k = len(feature_names)
    matrix = np.full((k, k), np.nan)
    for i in range(k):
        for j in range(i, k):
            r = (
                1.0
                if i == j and np.ptp(values[:, i]) > 0
                else _pair_correlation(values[:, i], values[:, j])[0]
            )
            matrix[i, j] = matrix[j, i] = r
    frame = pl.DataFrame({str(name): matrix[:, j] for j, name in enumerate(feature_names)})
    return frame.insert_column(
        0, pl.Series("feature", [str(n) for n in feature_names])
    ).with_columns(pl.selectors.float().fill_nan(None))


def _top_indices(shap_values: np.ndarray, k: int) -> np.ndarray:
    """平均|SHAP|が大きい順に上位 k 個の特徴量の番号。"""
    importance = np.nanmean(np.abs(shap_values), axis=0)
    return np.argsort(-np.nan_to_num(importance, nan=-1.0))[:k]


class ShapCorrelationBarDisplay:
    """平均|SHAP|の棒グラフを、特徴量の値と SHAP 値の相関の符号で色分けしたもの。

    赤: 値が大きいほど予測を押し上げる、青: 押し下げる、灰: 向きがはっきりしない
    （相関が小さい・計算できない）。棒の右に順位相関係数（Spearman）を表示する。

    Attributes:
        table: `shap_feature_correlation` の結果。
        figure_: 描画したFigure。
        ax_: 描画したAxes。
    """

    def __init__(self, table: pl.DataFrame) -> None:
        self.table = table

    @classmethod
    def from_shap(
        cls,
        shap_values: Any,
        data: Any,
        feature_names: Sequence[str],
        *,
        max_display: int = 20,
        ax: Any = None,
        title: str = "",
    ) -> Self:
        """SHAP値と特徴量の値から計算して描く。"""
        table = shap_feature_correlation(shap_values, data, feature_names)
        return cls(table).plot(ax=ax, max_display=max_display, title=title)

    def plot(self, ax: Any = None, *, max_display: int = 20, title: str = "") -> Self:
        """色分けした重要度の棒グラフを描く。"""
        ensure_japanese_font()
        top = self.table.head(max_display).reverse()
        fig, ax = new_axes(ax, figsize=(9, max(3, 0.38 * top.height + 1.5)))
        colors = [
            {"正": _POSITIVE, "負": _NEGATIVE}.get(d, _UNKNOWN) for d in top["direction"].to_list()
        ]
        bars = ax.barh(top["feature"].to_list(), top["mean_abs_shap"].to_list(), color=colors)
        for bar, r in zip(bars, top["spearman"].to_list(), strict=True):
            label = "相関なし" if r is None else f"r={r:+.2f}"
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {label}",
                va="center",
                fontsize=8,
            )
        ax.set_xlim(
            0, max(float(np.nanmax(top["mean_abs_shap"].to_numpy())), 1e-12) * 1.2
        )  # 棒グラフは0起点
        ax.set_xlabel("平均 |SHAP値|")
        ax.set_title(title or "SHAP重要度と効きの向き（特徴量の値とSHAP値の順位相関）")
        handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (_POSITIVE, _NEGATIVE, _UNKNOWN)]
        ax.legend(
            handles,
            ["値が大きいほど予測を上げる", "値が大きいほど予測を下げる", "向きがはっきりしない"],
            loc="lower right",
            fontsize=8,
        )
        self.figure_, self.ax_ = fig, ax
        return self


class ShapDependenceDisplay:
    """重要度上位の特徴量について「特徴量の値 vs SHAP 値」の散布図（dependence plot）を並べる。

    点の色は、その特徴量との相互作用が最も強そうな別の特徴量の値
    （`shap.utils.approximate_interactions` で選ぶ）。色の違いで SHAP 値が上下に分かれていれば、
    2つの特徴量の組み合わせで効き方が変わっている（相互作用がある）ことを示す。

    Attributes:
        features: 描いた特徴量名。
        color_features: 各特徴量の色に使った特徴量名（選べなかった場合はNone）。
        figure_: 描画したFigure。
        ax_: 描画したAxes。
    """

    figure_: plt.Figure
    ax_: Any

    def __init__(self, features: list[str], color_features: list[str | None]) -> None:
        self.features = features
        self.color_features = color_features

    @classmethod
    def from_shap(
        cls,
        shap_values: Any,
        data: Any,
        feature_names: Sequence[str],
        *,
        top_k: int = 6,
        max_points: int | None = 2000,
        ncols: int = 3,
        title: str = "",
    ) -> Self:
        """SHAP値と特徴量の値から、上位 `top_k` 個の特徴量の散布図を描く。"""
        ensure_japanese_font()
        values, x = _as_2d(shap_values), _as_2d(data)
        names = [str(n) for n in feature_names]
        top = _top_indices(values, min(top_k, len(names)))
        ncols = max(1, min(ncols, top.size))
        nrows = -(-top.size // ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True, squeeze=False
        )
        idx = subsample_index(values.shape[0], max_points)
        features, color_features = [], []
        for ax, j in zip(axes.flat, top, strict=False):
            color_j = cls._interaction_partner(int(j), values, x, names)
            points = {"s": 8, "alpha": 0.6}
            if color_j is None:
                ax.scatter(x[idx, j], values[idx, j], color=_NEGATIVE, **points)
            else:
                sc = ax.scatter(
                    x[idx, j], values[idx, j], c=x[idx, color_j], cmap="coolwarm", **points
                )
                fig.colorbar(sc, ax=ax, label=f"{names[color_j]} の値")
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_xlabel(f"{names[j]} の値")
            ax.set_ylabel(f"{names[j]} のSHAP値")
            r = _pair_correlation(x[:, j], values[:, j])[1]
            ax.set_title(
                f"{names[j]}（順位相関 r={r:+.2f}）" if np.isfinite(r) else names[j], fontsize=10
            )
            features.append(names[j])
            color_features.append(None if color_j is None else names[color_j])
        for ax in list(axes.flat)[top.size :]:
            ax.set_axis_off()
        fig.suptitle(
            title or "特徴量の値とSHAP値の関係（dependence plot。色は相互作用が強い特徴量）"
        )
        disp = cls(features, color_features)
        disp.figure_, disp.ax_ = fig, axes
        return disp

    @staticmethod
    def _interaction_partner(
        j: int, values: np.ndarray, x: np.ndarray, names: list[str]
    ) -> int | None:
        """特徴量 j との相互作用が最も強そうな別の特徴量の番号（見つからなければNone）。"""
        if len(names) < 2:
            return None
        try:
            order = shap.utils.approximate_interactions(j, values, x)
        except (ValueError, IndexError, FloatingPointError):
            # 欠損が多い・値の種類が少ない等で計算できない場合は色分けしない
            return None
        candidates = [int(k) for k in order if int(k) != j]
        return candidates[0] if candidates else None


class ShapValueCorrelationDisplay:
    """重要度上位の特徴量同士の SHAP 値の相関行列（ヒートマップ）。

    赤は SHAP 値が同じ向きに動く（似た情報を持つ・冗長な可能性）、青は逆向きに動く
    （打ち消し合う）組。値は Pearson の相関係数。

    Attributes:
        matrix: `shap_value_correlation` の結果（上位の特徴量のみ）。
        figure_: 描画したFigure。
        ax_: 描画したAxes。
    """

    def __init__(self, matrix: pl.DataFrame) -> None:
        self.matrix = matrix

    @classmethod
    def from_shap(
        cls,
        shap_values: Any,
        feature_names: Sequence[str],
        *,
        top_k: int = 15,
        ax: Any = None,
        title: str = "",
    ) -> Self:
        """SHAP値から、上位 `top_k` 個の特徴量の相関行列を計算して描く。"""
        values = _as_2d(shap_values)
        top = _top_indices(values, min(top_k, len(feature_names)))
        names = [str(feature_names[j]) for j in top]
        return cls(shap_value_correlation(values[:, top], names)).plot(ax=ax, title=title)

    def plot(self, ax: Any = None, *, title: str = "") -> Self:
        """ヒートマップを描く。"""
        ensure_japanese_font()
        names = self.matrix["feature"].to_list()
        k = len(names)
        fig, ax = new_axes(ax, figsize=(max(6, 0.6 * k + 3), max(5, 0.55 * k + 2)))
        values = self.matrix.select(names).to_numpy().astype(np.float64)
        image = ax.imshow(np.ma.masked_invalid(values), cmap="RdBu_r", vmin=-1, vmax=1)
        fig.colorbar(image, ax=ax, label="SHAP値の相関係数")
        ax.set_xticks(range(k), names, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(k), names, fontsize=8)
        if k <= 20:
            for i in range(k):
                for j in range(k):
                    if np.isfinite(values[i, j]):
                        color = "white" if abs(values[i, j]) > 0.6 else "black"
                        ax.text(
                            j,
                            i,
                            f"{values[i, j]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=color,
                        )
        ax.set_title(title or "特徴量同士のSHAP値の相関（重要度上位）")
        self.figure_, self.ax_ = fig, ax
        return self


class ShapScatterMatrixDisplay:
    """重要度上位の特徴量について、SHAP 値同士の散布図行列。

    下三角: 特徴量 i と j の SHAP 値の散布図（相関係数を表示）、対角: SHAP 値のヒストグラム、
    上三角: 空白。点が右上がりに並べば2つの特徴量は同じ向きに効き、右下がりなら打ち消し合う。

    Attributes:
        features: 描いた特徴量名。
        figure_: 描画したFigure。
        ax_: 描画したAxes（K × K）。
    """

    figure_: plt.Figure
    ax_: Any

    def __init__(self, features: list[str]) -> None:
        self.features = features

    @classmethod
    def from_shap(
        cls,
        shap_values: Any,
        feature_names: Sequence[str],
        *,
        top_k: int = 5,
        max_points: int | None = 2000,
        title: str = "",
    ) -> Self:
        """SHAP値から、上位 `top_k` 個の特徴量の散布図行列を描く。"""
        ensure_japanese_font()
        values = _as_2d(shap_values)
        top = _top_indices(values, min(top_k, len(feature_names)))
        names = [str(feature_names[j]) for j in top]
        k = top.size
        fig, axes = plt.subplots(
            k, k, figsize=(2.6 * k + 1, 2.6 * k + 1), constrained_layout=True, squeeze=False
        )
        idx = subsample_index(values.shape[0], max_points)
        for r in range(k):
            for c in range(k):
                ax = axes[r, c]
                vr, vc = values[:, top[r]], values[:, top[c]]
                if r == c:
                    ax.hist(vr[np.isfinite(vr)], bins=30, color=_NEGATIVE, alpha=0.7)
                elif r > c:
                    ax.scatter(vc[idx], vr[idx], s=5, alpha=0.4, color=_NEGATIVE)
                    corr = _pair_correlation(vc, vr)[0]
                    if np.isfinite(corr):
                        ax.text(
                            0.03,
                            0.95,
                            f"r={corr:+.2f}",
                            transform=ax.transAxes,
                            va="top",
                            fontsize=9,
                            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                        )
                    ax.axhline(0, color="gray", linewidth=0.5)
                    ax.axvline(0, color="gray", linewidth=0.5)
                else:
                    ax.set_axis_off()
                    continue
                if r == k - 1:
                    ax.set_xlabel(f"{names[c]}\nのSHAP値", fontsize=8)
                if c == 0 and r > 0:
                    ax.set_ylabel(f"{names[r]}\nのSHAP値", fontsize=8)
                ax.tick_params(labelsize=7)
        fig.suptitle(title or "特徴量同士のSHAP値の散布図（下三角: 散布図、対角: SHAP値の分布）")
        disp = cls(names)
        disp.figure_, disp.ax_ = fig, axes
        return disp
