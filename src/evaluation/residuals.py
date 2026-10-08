"""回帰モデルの残差の可視化（残差分布・残差プロット・正規Q-Qプロット・残差のACF/PACF）。

残差は `実測値 − 予測値`（sklearn の `PredictionErrorDisplay` と同じ向き）。
良いモデルの残差は「平均0付近・予測値の大きさによらず一定のばらつき・正規分布に近い・
自己相関が無い」ことが期待される。各図はそのどれが崩れているかを見るためのもの。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from sklearn.metrics import PredictionErrorDisplay

from eda.time_series_eda import compute_acf, compute_pacf
from evaluation._common import as_1d_float, finite_pair, new_axes, subsample_index
from util.plotting import ensure_japanese_font


def residual_summary(y_true: Any, y_pred: Any) -> dict[str, float]:
    """残差の要約統計（件数・平均・標準偏差・歪度・尖度（正規分布で0）・MAE・RMSE）。"""
    t, p = finite_pair(y_true, y_pred)
    r = t - p
    return {
        "n": float(r.size),
        "mean": float(r.mean()),
        "std": float(r.std(ddof=1)) if r.size > 1 else float("nan"),
        "skewness": float(stats.skew(r)) if r.size > 2 else float("nan"),
        "excess_kurtosis": float(stats.kurtosis(r)) if r.size > 3 else float("nan"),
        "mae": float(np.abs(r).mean()),
        "rmse": float(np.sqrt((r**2).mean())),
    }


class ResidualDistributionDisplay:
    """残差のヒストグラムと、同じ平均・標準偏差の正規分布の密度。

    Attributes:
        residuals: 残差（欠損を除く）。
        summary: `residual_summary` の結果。
        figure_: 描画したFigure。
        ax_: 描画したAxes。
    """

    def __init__(self, residuals: np.ndarray, summary: dict[str, float]) -> None:
        self.residuals = residuals
        self.summary = summary

    @classmethod
    def from_predictions(
        cls, y_true: Any, y_pred: Any, *, ax: Any = None, bins: int | str = "auto"
    ) -> Self:
        """実測値と予測値から残差を計算して描く。"""
        t, p = finite_pair(y_true, y_pred)
        return cls.from_residuals(t - p, ax=ax, bins=bins)

    @classmethod
    def from_residuals(cls, residuals: Any, *, ax: Any = None, bins: int | str = "auto") -> Self:
        """残差（実測値 − 予測値）から描く（欠損は除く）。"""
        r = as_1d_float(residuals)
        r = r[np.isfinite(r)]
        # residual_summary は (実測値, 予測値) を受け取るため、予測値0として残差をそのまま渡す
        return cls(r, residual_summary(r, np.zeros_like(r))).plot(ax=ax, bins=bins)

    def plot(self, ax: Any = None, *, bins: int | str = "auto") -> Self:
        """残差分布を描く。"""
        ensure_japanese_font()
        fig, ax = new_axes(ax, figsize=(8, 5))
        r, s = self.residuals, self.summary
        ax.hist(r, bins=bins, density=True, alpha=0.6, color="#4c72b0", label="残差")
        if np.isfinite(s["std"]) and s["std"] > 0:
            grid = np.linspace(r.min(), r.max(), 200)
            ax.plot(
                grid, stats.norm.pdf(grid, s["mean"], s["std"]), color="#c44e52", label="正規分布"
            )
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("残差（実測値 − 予測値）")
        ax.set_ylabel("密度")
        ax.set_title(
            f"残差分布（平均={s['mean']:.3g}, 標準偏差={s['std']:.3g}, "
            f"歪度={s['skewness']:.2f}, 尖度={s['excess_kurtosis']:.2f}）"
        )
        ax.legend()
        self.figure_, self.ax_ = fig, ax
        return self


class ResidualPlotDisplay:
    """2枚組の残差プロット: 残差 vs 予測値 ／ 実測値 vs 予測値。

    描画には sklearn の `PredictionErrorDisplay` を使う。点が多い場合は `max_points` 点に間引く。

    Attributes:
        y_true: 実測値（欠損を除く）。
        y_pred: 予測値（欠損を除く）。
        displays_: sklearn の `PredictionErrorDisplay`（残差, 実測）の組。
        figure_: 描画したFigure。
        ax_: 描画したAxes（2個）。
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        self.y_true = y_true
        self.y_pred = y_pred

    @classmethod
    def from_predictions(
        cls, y_true: Any, y_pred: Any, *, ax: Any = None, max_points: int | None = 5000
    ) -> Self:
        """実測値と予測値から描く。"""
        t, p = finite_pair(y_true, y_pred)
        return cls(t, p).plot(ax=ax, max_points=max_points)

    def plot(self, ax: Any = None, *, max_points: int | None = 5000) -> Self:
        """2枚組の残差プロットを描く（`ax` を渡す場合は2個の並び）。"""
        ensure_japanese_font()
        fig, axes = new_axes(ax, ncols=2, figsize=(13, 5.5))
        idx = subsample_index(self.y_true.size, max_points)
        t, p = self.y_true[idx], self.y_pred[idx]
        scatter = {"alpha": 0.4, "s": 10}
        residual = PredictionErrorDisplay.from_predictions(
            t, p, kind="residual_vs_predicted", ax=axes.flat[0], scatter_kwargs=scatter
        )
        actual = PredictionErrorDisplay.from_predictions(
            t, p, kind="actual_vs_predicted", ax=axes.flat[1], scatter_kwargs=scatter
        )
        axes.flat[0].set_title("残差 vs 予測値（0の周りに一様に散らばるのが理想）")
        axes.flat[0].set_xlabel("予測値")
        axes.flat[0].set_ylabel("残差（実測値 − 予測値）")
        axes.flat[1].set_title("実測値 vs 予測値（対角線に近いほど良い）")
        axes.flat[1].set_xlabel("予測値")
        axes.flat[1].set_ylabel("実測値")
        self.displays_ = (residual, actual)
        self.figure_, self.ax_ = fig, axes
        return self


class QQPlotDisplay:
    """標準化した残差の正規Q-Qプロット。

    点が対角線から外れるほど、残差の分布が正規分布から離れている（両端の外れは裾の重さ）。

    Attributes:
        theoretical: 正規分布の理論分位点。
        ordered: 標準化残差を小さい順に並べた値。
        r: 当てはめ直線の相関係数（1に近いほど正規分布に近い）。
        figure_: 描画したFigure。
        ax_: 描画したAxes。
    """

    def __init__(self, theoretical: np.ndarray, ordered: np.ndarray, r: float) -> None:
        self.theoretical = theoretical
        self.ordered = ordered
        self.r = r

    @classmethod
    def from_predictions(cls, y_true: Any, y_pred: Any, *, ax: Any = None) -> Self:
        """実測値と予測値から残差を計算し、標準化してQ-Qプロットを描く。"""
        t, p = finite_pair(y_true, y_pred)
        return cls.from_residuals(t - p, ax=ax)

    @classmethod
    def from_residuals(cls, residuals: Any, *, ax: Any = None) -> Self:
        """残差（実測値 − 予測値）を標準化してQ-Qプロットを描く（欠損は除く）。"""
        residuals = as_1d_float(residuals)
        residuals = residuals[np.isfinite(residuals)]
        std = residuals.std(ddof=1)
        standardized = (residuals - residuals.mean()) / std if std > 0 else residuals * 0.0
        (theoretical, ordered), (_, _, r) = stats.probplot(standardized, dist="norm")
        return cls(np.asarray(theoretical), np.asarray(ordered), float(r)).plot(ax=ax)

    def plot(self, ax: Any = None) -> Self:
        """Q-Qプロットを描く。"""
        ensure_japanese_font()
        fig, ax = new_axes(ax, figsize=(6, 6))
        ax.scatter(self.theoretical, self.ordered, s=8, alpha=0.5, color="#4c72b0")
        lims = [
            min(self.theoretical.min(), self.ordered.min()),
            max(self.theoretical.max(), self.ordered.max()),
        ]
        ax.plot(lims, lims, color="#c44e52", linewidth=1, label="正規分布なら乗る線")
        ax.set_xlabel("正規分布の理論分位点")
        ax.set_ylabel("標準化残差の分位点")
        ax.set_title(f"正規Q-Qプロット（相関係数 r={self.r:.3f}）")
        ax.legend(loc="upper left")
        self.figure_, self.ax_ = fig, ax
        return self


def plot_residuals_by_group(
    kind: Literal["distribution", "qq"],
    groups: Mapping[str, tuple[Any, Any]],
    *,
    ncols: int = 2,
) -> tuple[plt.Figure, dict[str, ResidualDistributionDisplay | QQPlotDisplay]]:
    """系列（グループ）ごとに、残差分布または正規Q-Qプロットをパネルに並べて描く。

    尺度の違う複数の系列の残差を1つの分布にまとめると、系列間の差が裾の重さのように見えて
    分布の形が正しく読めない。そのため系列ごとに分けて描く。

    Args:
        kind: `distribution`（残差分布）または `qq`（正規Q-Qプロット）。
        groups: 系列名 → (実測値, 予測値)。
        ncols: 1行に並べるパネル数。

    Returns:
        (Figure, 系列名 → 描画したDisplay)。
    """
    ensure_japanese_font()
    names = list(groups)
    ncols = max(1, min(ncols, len(names)))
    nrows = -(-len(names) // ncols)  # 切り上げ
    size = (6.5 * ncols, 4.5 * nrows) if kind == "distribution" else (5.5 * ncols, 5.2 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=size, constrained_layout=True, squeeze=False)
    displays: dict[str, ResidualDistributionDisplay | QQPlotDisplay] = {}
    for ax, name in zip(axes.flat, names, strict=False):
        y_true, y_pred = groups[name]
        display: ResidualDistributionDisplay | QQPlotDisplay
        if kind == "distribution":
            display = ResidualDistributionDisplay.from_predictions(y_true, y_pred, ax=ax)
        else:
            display = QQPlotDisplay.from_predictions(y_true, y_pred, ax=ax)
        # 各パネルの見出しの先頭に系列名を付ける
        ax.set_title(f"{name}: {ax.get_title()}", fontsize=10)
        displays[name] = display
    for ax in list(axes.flat)[len(names) :]:
        ax.set_axis_off()
    return fig, displays


class ResidualCorrelogramDisplay:
    """残差の自己相関係数（ACF）と偏自己相関係数（PACF）のコレログラム。

    時系列モデルでは、残差に自己相関が残っていれば「まだ予測に使える過去の情報がある」ことを示す。
    残差は **時刻順** に並べて渡すこと。複数の系列がある場合は系列ごとに1段ずつ描く。
    点線は無相関の場合の95%信頼区間（±1.96/√n）。計算は `eda.time_series_eda` の関数を使う。

    Attributes:
        acf: 系列名 → ACF（ラグ0から）。
        pacf: 系列名 → PACF（ラグ0から）。
        n_obs: 系列名 → 計算に使った点数。
        figure_: 描画したFigure。
        ax_: 描画したAxes（系列数 × 2）。
    """

    def __init__(
        self,
        acf: dict[str, np.ndarray | None],
        pacf: dict[str, np.ndarray | None],
        n_obs: dict[str, int],
    ) -> None:
        self.acf = acf
        self.pacf = pacf
        self.n_obs = n_obs

    @classmethod
    def from_residuals(
        cls, residuals: Mapping[str, Any] | Any, *, nlags: int = 40, ax: Any = None
    ) -> Self:
        """時刻順の残差（系列名 → 残差、または1系列の残差）から計算して描く。"""
        series = residuals if isinstance(residuals, Mapping) else {"残差": residuals}
        acf, pacf, n_obs = {}, {}, {}
        for name, values in series.items():
            s = pl.Series(np.asarray(values, dtype=np.float64)).fill_nan(None).drop_nulls()
            acf[str(name)] = compute_acf(s, nlags=nlags)
            pacf[str(name)] = compute_pacf(s, nlags=nlags)
            n_obs[str(name)] = s.len()
        return cls(acf, pacf, n_obs).plot(ax=ax)

    @classmethod
    def from_predictions(cls, y_true: Any, y_pred: Any, *, nlags: int = 40, ax: Any = None) -> Self:
        """時刻順に並んだ実測値・予測値（1系列）から残差を計算して描く。"""
        residuals = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
        return cls.from_residuals(residuals, nlags=nlags, ax=ax)

    def plot(self, ax: Any = None) -> Self:
        """コレログラムを描く（行: 系列, 列: ACF / PACF）。"""
        ensure_japanese_font()
        names = list(self.acf)
        fig, axes = new_axes(ax, ncols=2, nrows=len(names), figsize=(13, 3.2 * len(names) + 0.8))
        axes = np.asarray(axes, dtype=object).reshape(len(names), 2)
        for row, name in enumerate(names):
            for col, (label, values) in enumerate(
                (("ACF", self.acf[name]), ("PACF", self.pacf[name]))
            ):
                a = axes[row, col]
                if values is None:
                    a.text(0.5, 0.5, "データ不足", ha="center", va="center", transform=a.transAxes)
                    a.set_axis_off()
                    continue
                lags = np.arange(values.size)
                a.vlines(lags, 0, values, color="#4c72b0")
                a.scatter(lags, values, s=10, color="#4c72b0")
                a.axhline(0, color="black", linewidth=0.8)
                band = 1.96 / np.sqrt(max(self.n_obs[name], 1))
                a.axhline(band, color="gray", linestyle="--", linewidth=1)
                a.axhline(-band, color="gray", linestyle="--", linewidth=1)
                a.set_title(f"{name}: 残差の{label}（n={self.n_obs[name]:,}）")
                a.set_xlabel("ラグ")
                a.set_ylabel(label)
        self.figure_, self.ax_ = fig, axes
        return self
