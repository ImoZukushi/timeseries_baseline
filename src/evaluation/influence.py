"""Leverage と Cook の距離（影響の大きいサンプルの診断）。

- **Leverage** `h_i`: ハット行列 `H = X (XᵀX)⁻¹ Xᵀ` の対角成分。特徴量の空間で他のサンプルから
  離れている（外れた組み合わせの）サンプルほど大きい。合計は特徴量の数（切片込み、行列のランク）に等しい。
- **標準化残差** `r_i = e_i / (s √(1 − h_i))`（`s² = Σe²/(n − p)`）。
- **Cook の距離** `D_i = r_i² / p × h_i / (1 − h_i)`。そのサンプルを除いて学習し直すと予測が
  どれだけ変わるかの目安。`D_i > 4/n` や `D_i > 0.5` が注意の目安としてよく使われる。

これらは本来、線形回帰の診断量である。ここではモデルに入力した特徴量行列（前処理後）から
Leverage を、任意のモデルの残差から Cook の距離を計算する。線形回帰では `statsmodels` の
`OLSInfluence` と一致し、非線形モデル（GBDT等）では「特徴量が外れていて、かつ誤差が大きい
サンプル」を見つける目安として使う。
"""

from __future__ import annotations

from typing import Any, Self

import numpy as np
import polars as pl

from evaluation._common import as_1d_float, new_axes, subsample_index
from util.plotting import ensure_japanese_font


def _design_matrix(X: Any) -> np.ndarray:
    """特徴量行列を float に変換し、欠損は列の中央値で埋め、切片の列を加える。"""
    arr = np.asarray(X, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    # ハット行列は欠損を扱えないため、列の中央値で埋める（全て欠損の列は0）
    medians = np.nanmedian(np.where(np.isfinite(arr), arr, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    arr = np.where(np.isfinite(arr), arr, medians)
    return np.column_stack([np.ones(arr.shape[0]), arr])


def compute_influence(X: Any, residuals: Any) -> pl.DataFrame:
    """Leverage・標準化残差・Cook の距離をサンプルごとに計算する。

    特異値分解でハット行列の対角を求めるため、特徴量が互いに線形従属（多重共線性）でも計算できる
    （その場合 p は行列のランクになる）。

    Args:
        X: 特徴量行列（n × 特徴量数。モデルに入力した前処理後のもの）。
        residuals: 残差（長さ n）。

    Returns:
        列 `row`（入力の行番号）, `leverage`, `residual`, `standardized_residual`,
        `cooks_distance` の表。

    Raises:
        ValueError: 行数が合わない場合、または行数が特徴量数（ランク）以下の場合。
    """
    design = _design_matrix(X)
    e = as_1d_float(residuals)
    n = design.shape[0]
    if e.size != n:
        raise ValueError(f"X と residuals の行数が違います: {n} != {e.size}")
    u, singular, _ = np.linalg.svd(design, full_matrices=False)
    tol = singular.max() * max(design.shape) * np.finfo(np.float64).eps
    rank = int((singular > tol).sum())
    if n <= rank:
        raise ValueError(f"行数（{n}）が特徴量のランク（{rank}）以下のため計算できません")
    # ハット行列の対角 = 列空間の正規直交基底 U の各行の二乗和
    leverage = (u[:, :rank] ** 2).sum(axis=1)
    s2 = float((e**2).sum() / (n - rank))
    with np.errstate(divide="ignore", invalid="ignore"):
        standardized = e / np.sqrt(s2 * (1.0 - leverage))
        cooks = standardized**2 / rank * leverage / (1.0 - leverage)
    return pl.DataFrame(
        {
            "row": np.arange(n),
            "leverage": leverage,
            "residual": e,
            "standardized_residual": standardized,
            "cooks_distance": cooks,
        }
    ).with_columns(pl.selectors.float().fill_nan(None))


class InfluenceDisplay:
    """Leverage vs 標準化残差（Cook の距離の等高線つき）と、Cook の距離の棒グラフ。

    Attributes:
        influence: `compute_influence` の結果。
        n_params: 計算に使った特徴量のランク（切片込み）。
        figure_: 描画したFigure。
        ax_: 描画したAxes（2個）。
    """

    def __init__(self, influence: pl.DataFrame, n_params: int) -> None:
        self.influence = influence
        self.n_params = n_params

    @classmethod
    def from_predictions(
        cls,
        X: Any,
        y_true: Any,
        y_pred: Any,
        *,
        ax: Any = None,
        max_points: int | None = 5000,
        n_labels: int = 5,
    ) -> Self:
        """特徴量行列・実測値・予測値から計算して描く（実測値・予測値の欠損行は除く）。"""
        t, p = as_1d_float(y_true), as_1d_float(y_pred)
        mask = np.isfinite(t) & np.isfinite(p)
        X_arr = np.asarray(X, dtype=np.float64)[mask]
        influence = compute_influence(X_arr, t[mask] - p[mask])
        # 元の行番号に戻す（欠損行を除いたため）
        influence = influence.with_columns(pl.Series("row", np.flatnonzero(mask)))
        n_params = int(round(float(influence["leverage"].sum())))
        return cls(influence, n_params).plot(ax=ax, max_points=max_points, n_labels=n_labels)

    def top(self, k: int = 20) -> pl.DataFrame:
        """Cook の距離が大きい順に上位 `k` 行を返す。"""
        return self.influence.sort("cooks_distance", descending=True, nulls_last=True).head(k)

    def plot(self, ax: Any = None, *, max_points: int | None = 5000, n_labels: int = 5) -> Self:
        """2枚組の図を描く（`ax` を渡す場合は2個の並び）。"""
        ensure_japanese_font()
        fig, axes = new_axes(ax, ncols=2, figsize=(14, 5.5))
        inf = self.influence.drop_nulls(["leverage", "standardized_residual", "cooks_distance"])
        n, p = inf.height, max(self.n_params, 1)
        top = inf.sort("cooks_distance", descending=True).head(n_labels)
        # 散布図は間引くが、Cook の距離の上位は必ず描く
        idx = subsample_index(n, max_points)
        shown = pl.concat([inf[idx], top]).unique("row", keep="first")

        a = axes.flat[0]
        a.scatter(shown["leverage"], shown["standardized_residual"], s=10, alpha=0.4)
        h_max = float(inf["leverage"].to_numpy().max()) if n else 1.0
        h = np.linspace(1e-4, min(0.999, max(h_max * 1.05, 1e-3)), 200)
        for d, style in ((0.5, "--"), (1.0, "-")):
            bound = np.sqrt(d * p * (1 - h) / h)
            a.plot(h, bound, color="#c44e52", linestyle=style, linewidth=1, label=f"Cookの距離={d}")
            a.plot(h, -bound, color="#c44e52", linestyle=style, linewidth=1)
        r_abs = float(np.nanmax(np.abs(inf["standardized_residual"].to_numpy()))) if n else 1.0
        a.set_ylim(-max(r_abs, 3) * 1.1, max(r_abs, 3) * 1.1)
        for row in top.iter_rows(named=True):
            a.annotate(str(row["row"]), (row["leverage"], row["standardized_residual"]), fontsize=8)
        a.axhline(0, color="gray", linewidth=0.8)
        a.set_xlabel("Leverage（ハット行列の対角）")
        a.set_ylabel("標準化残差")
        a.set_title("Leverage vs 標準化残差（右上・右下ほど影響が大きい）")
        a.legend(loc="best")

        b = axes.flat[1]
        b.vlines(inf["row"], 0, inf["cooks_distance"], color="#4c72b0", linewidth=0.8)
        threshold = 4 / n if n else 0.0
        b.axhline(
            threshold,
            color="#c44e52",
            linestyle="--",
            linewidth=1,
            label=f"目安 4/n={threshold:.2g}",
        )
        for row in top.iter_rows(named=True):
            b.annotate(str(row["row"]), (row["row"], row["cooks_distance"]), fontsize=8)
        b.set_ylim(bottom=0)
        b.set_xlabel("行番号")
        b.set_ylabel("Cookの距離")
        b.set_title(
            f"Cookの距離（目安を超える行: {int((inf['cooks_distance'] > threshold).sum())}件）"
        )
        b.legend(loc="upper right")
        self.figure_, self.ax_ = fig, axes
        return self
