"""時系列モデルの残差診断（残差の時系列・検定・診断パネル）。

時系列モデルの残差（`実測値 − 予測値`、**時刻順**）が「白色雑音」に近いかを調べる。

| 観点 | 図 | 検定（帰無仮説） |
|---|---|---|
| 偏り・ばらつきの時間変化 | 残差の折れ線（`ResidualTimeSeriesDisplay`） | － |
| 自己相関が残っていないか | ACF/PACF（`ResidualCorrelogramDisplay`） | Ljung-Box（自己相関なし） |
| 正規分布に近いか | ヒストグラム・Q-Q（`residuals` の Display） | Jarque-Bera（正規分布） |
| 定常か（トレンド・水準の変化が残っていないか） | － | ADF（単位根あり）＋ KPSS（定常） |

`TimeSeriesResidualDiagnosticsDisplay` は1系列の診断を1枚（折れ線・ヒストグラム・Q-Q・ACF・PACF・
検定結果の表）にまとめる。検定は欠損を除いた時刻順の残差で行い、点数が足りない・値が一定などで
計算できない場合は例外にせず欠損（NaN）を返す。

注意:
- 機械学習モデルでは推定したパラメータ数が定義できないため、Ljung-Box の自由度補正（`model_df`）は
  既定で 0（ARIMA等で使う場合は、ARとMAの次数の和を渡す）。
- 多段先（再帰）予測の残差は誤差が蓄積するため、自己相関があるのが自然。モデルの当てはまりの
  診断には1期先予測の残差を使う。
- 複数の区間（CVのfoldなど）をつないだ残差では、区間の境目も連続として扱われる。
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Self

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss

from eda.time_series_eda import break_line_at_gaps
from evaluation._common import as_1d_float, new_axes
from evaluation.residuals import (
    QQPlotDisplay,
    ResidualCorrelogramDisplay,
    ResidualDistributionDisplay,
)
from util.plotting import ensure_japanese_font

# 検定に必要な最小の点数（これ未満は計算しない）
_MIN_OBS = 8

STATIONARY = "定常"
NON_STATIONARY = "非定常（単位根の疑い）"
INCONCLUSIVE_WEAK = "判定が分かれる（データ不足・検出力不足の可能性）"
INCONCLUSIVE_BREAK = "判定が分かれる（構造変化・差分定常の可能性）"
NOT_COMPUTED = "計算できない"

_NAN = float("nan")


def _clean(residuals: Any) -> np.ndarray:
    """欠損を除いた1次元の残差（時刻順はそのまま）。"""
    r = as_1d_float(residuals)
    return r[np.isfinite(r)]


def _testable(r: np.ndarray) -> bool:
    return r.size >= _MIN_OBS and float(np.ptp(r)) > 0


def default_ljung_box_lags(n: int, seasonal_period: int | None = None) -> list[int]:
    """Ljung-Box検定の既定のラグ: `min(10, n//5)` と、`n//2` 未満なら季節周期。"""
    lags = [max(1, min(10, n // 5))]
    if seasonal_period is not None and lags[0] < seasonal_period < n // 2:
        lags.append(int(seasonal_period))
    return lags


def ljung_box_test(
    residuals: Any,
    lags: Sequence[int] | None = None,
    *,
    seasonal_period: int | None = None,
    model_df: int = 0,
) -> pl.DataFrame:
    """Ljung-Box検定（帰無仮説: 指定ラグまで自己相関がない）。

    Args:
        residuals: 時刻順の残差。
        lags: 検定するラグ。Noneなら `default_ljung_box_lags`。点数の半分以上のラグは除く。
        seasonal_period: 既定のラグに加える季節周期。
        model_df: 自由度から差し引くモデルのパラメータ数（機械学習モデルでは0）。

    Returns:
        列 `lag`, `statistic`, `p_value` の表（計算できない場合は0行）。
    """
    r = _clean(residuals)
    empty = pl.DataFrame(schema={"lag": pl.Int64, "statistic": pl.Float64, "p_value": pl.Float64})
    if not _testable(r):
        return empty
    candidates = default_ljung_box_lags(r.size, seasonal_period) if lags is None else lags
    # 自由度（ラグ − model_df）が正で、点数の半分未満のラグだけを使う
    valid = sorted({int(k) for k in candidates if model_df < int(k) < r.size // 2})
    if not valid:
        return empty
    result = acorr_ljungbox(r, lags=valid, model_df=model_df)
    return pl.DataFrame(
        {
            "lag": valid,
            "statistic": result["lb_stat"].to_numpy(dtype=np.float64),
            "p_value": result["lb_pvalue"].to_numpy(dtype=np.float64),
        }
    )


def jarque_bera_test(residuals: Any) -> dict[str, float]:
    """Jarque-Bera検定（帰無仮説: 正規分布。歪度0・尖度0（正規分布との差）かどうかを見る）。

    Returns:
        `statistic`, `p_value`, `skewness`, `excess_kurtosis`（計算できない場合はNaN）。
    """
    r = _clean(residuals)
    if not _testable(r):
        return {"statistic": _NAN, "p_value": _NAN, "skewness": _NAN, "excess_kurtosis": _NAN}
    result = stats.jarque_bera(r)
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "skewness": float(stats.skew(r)),
        "excess_kurtosis": float(stats.kurtosis(r)),
    }


def unit_root_test(
    residuals: Any, *, regression: str = "c", alpha: float = 0.05
) -> dict[str, float | int | str]:
    """ADF検定とKPSS検定を組み合わせた定常性の判定。

    - ADF（拡張Dickey-Fuller）: 帰無仮説は「単位根あり（非定常）」。ラグ次数はAICで選ぶ。
    - KPSS: 帰無仮説は「定常」。p値は統計表の範囲（0.01〜0.1）で打ち切られる。

    帰無仮説が逆の2つの検定を組み合わせ、`conclusion` を次のように決める。

    | ADF | KPSS | 判定 |
    |---|---|---|
    | 棄却 | 棄却せず | 定常 |
    | 棄却せず | 棄却 | 非定常（単位根の疑い） |
    | 棄却せず | 棄却せず | 判定が分かれる（データ不足・検出力不足の可能性） |
    | 棄却 | 棄却 | 判定が分かれる（構造変化・差分定常の可能性） |

    Args:
        residuals: 時刻順の残差。
        regression: 検定式の確定項。`c`（定数）または `ct`（定数＋線形トレンド）。
        alpha: 有意水準。

    Returns:
        `adf_statistic`, `adf_p_value`, `adf_lags`, `kpss_statistic`, `kpss_p_value`, `conclusion`。
    """
    r = _clean(residuals)
    result: dict[str, float | int | str] = {
        "adf_statistic": _NAN,
        "adf_p_value": _NAN,
        "adf_lags": -1,
        "kpss_statistic": _NAN,
        "kpss_p_value": _NAN,
        "conclusion": NOT_COMPUTED,
    }
    if not _testable(r) or r.size < 20:
        return result
    try:
        # 結果オブジェクトで受け取る（statsmodels の将来の既定。タプルの並びに依存しない）
        adf = adfuller(r, regression=regression, autolag="AIC", result_object=True)
        with warnings.catch_warnings():
            # p値が統計表の範囲外のときの警告（値は端で打ち切られる。docstringに明記）
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_result = kpss(r, regression=regression, nlags="auto", result_object=True)
    except (ValueError, np.linalg.LinAlgError):
        return result
    adf_stat, adf_p, adf_lags = adf.statistic, adf.pvalue, adf.lags
    kpss_stat, kpss_p = kpss_result.statistic, kpss_result.pvalue
    adf_reject, kpss_reject = adf_p < alpha, kpss_p < alpha
    if adf_reject and not kpss_reject:
        conclusion = STATIONARY
    elif not adf_reject and kpss_reject:
        conclusion = NON_STATIONARY
    elif not adf_reject:
        conclusion = INCONCLUSIVE_WEAK
    else:
        conclusion = INCONCLUSIVE_BREAK
    result.update(
        adf_statistic=float(adf_stat),
        adf_p_value=float(adf_p),
        adf_lags=int(adf_lags),
        kpss_statistic=float(kpss_stat),
        kpss_p_value=float(kpss_p),
        conclusion=conclusion,
    )
    return result


def _as_series_mapping(residuals: Mapping[str, Any] | Any, default: str = "残差") -> dict[str, Any]:
    if isinstance(residuals, Mapping):
        return {str(k): v for k, v in residuals.items()}
    return {default: residuals}


def residual_tests(
    residuals: Mapping[str, Any] | Any,
    *,
    lags: Sequence[int] | None = None,
    seasonal_period: int | None = None,
    regression: str = "c",
    alpha: float = 0.05,
    model_df: int = 0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """系列ごとに Ljung-Box・Jarque-Bera・ADF・KPSS をまとめて行う。

    Args:
        residuals: 系列名 → 時刻順の残差（または1系列の残差）。
        lags: Ljung-Boxのラグ（Noneなら系列ごとに既定値）。
        seasonal_period: Ljung-Boxの既定のラグに加える季節周期。
        regression: 単位根検定の確定項（`c` / `ct`）。
        alpha: 判定に使う有意水準。
        model_df: Ljung-Boxの自由度補正。

    Returns:
        (要約表, Ljung-Box表)。
        要約表は1系列1行。Ljung-Boxは最大のラグの結果を載せ、判定列 `autocorrelation`（あり/なし）・
        `normality`（棄却/棄却せず）・`stationarity`（単位根検定の判定）を持つ。
        Ljung-Box表は列 `series`, `lag`, `statistic`, `p_value`（全ラグ）。
    """
    summaries, lb_frames = [], []
    for name, values in _as_series_mapping(residuals).items():
        r = _clean(values)
        lb = ljung_box_test(r, lags, seasonal_period=seasonal_period, model_df=model_df)
        jb = jarque_bera_test(r)
        ur = unit_root_test(r, regression=regression, alpha=alpha)
        lb_last = lb.row(-1, named=True) if lb.height else None
        lb_p = _NAN if lb_last is None else float(lb_last["p_value"])
        summaries.append(
            {
                "series": name,
                "n": int(r.size),
                "ljung_box_lag": None if lb_last is None else int(lb_last["lag"]),
                "ljung_box_statistic": _NAN if lb_last is None else float(lb_last["statistic"]),
                "ljung_box_p_value": lb_p,
                "jarque_bera_statistic": jb["statistic"],
                "jarque_bera_p_value": jb["p_value"],
                "skewness": jb["skewness"],
                "excess_kurtosis": jb["excess_kurtosis"],
                **ur,
                "autocorrelation": _verdict(lb_p, alpha, "あり", "なし"),
                "normality": _verdict(jb["p_value"], alpha, "棄却", "棄却せず"),
                "stationarity": ur["conclusion"],
            }
        )
        lb_frames.append(lb.insert_column(0, pl.Series("series", [name] * lb.height, pl.String)))
    summary = (
        pl.DataFrame(summaries).drop("conclusion").with_columns(pl.selectors.float().fill_nan(None))
    )
    return summary, pl.concat(lb_frames)


def _verdict(p_value: float, alpha: float, reject: str, keep: str) -> str:
    if not np.isfinite(p_value):
        return NOT_COMPUTED
    return reject if p_value < alpha else keep


# --- 図 ------------------------------------------------------------------------------------


def _line_values(
    time: np.ndarray, values: np.ndarray, segments: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """折れ線用の値（区間の境目・時刻の大きな空白で線を切る）と、区間の境目の位置を返す。"""
    line = break_line_at_gaps(time, values)
    boundaries = np.zeros(0, dtype=np.int64)
    if segments is not None and segments.size > 1:
        boundaries = np.flatnonzero(segments[1:] != segments[:-1]) + 1
        line[boundaries] = np.nan
    return line, boundaries


class ResidualTimeSeriesDisplay:
    """残差の時系列（折れ線）。系列ごとに1段ずつ描く。

    0の線・移動平均（偏りの時間変化）・±2σの帯（残差全体の標準偏差）を重ねる。区間（CVのfoldなど）の
    境目と時刻の大きな空白では線を切り、境目に縦の点線を引く。移動平均が0から離れる期間は予測が
    偏っており、振れ幅が時期によって変わるならばらつき（分散）が一定でない。

    Attributes:
        series: 系列名 → (時刻, 残差, 区間ID（無ければNone）)。
        rolling_window: 移動平均の窓幅。
        figure_: 描画したFigure。
        ax_: 描画したAxes（1系列ならAxes、複数系列なら系列数個の配列）。
    """

    figure_: plt.Figure
    ax_: Any

    def __init__(
        self,
        series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
        rolling_window: int | None = None,
    ) -> None:
        self.series = series
        self.rolling_window = rolling_window

    @classmethod
    def from_residuals(
        cls,
        residuals: Mapping[str, Any] | Any,
        *,
        time: Mapping[str, Any] | Any = None,
        segments: Mapping[str, Any] | Any = None,
        rolling_window: int | None = None,
        ax: Any = None,
    ) -> Self:
        """時刻順の残差から描く。

        Args:
            residuals: 系列名 → 残差（または1系列の残差）。
            time: 各残差の時刻（`residuals` と同じ形。Noneなら並び順の番号）。
            segments: 各残差の区間ID（foldなど。値が変わる所で線を切る）。
            rolling_window: 移動平均の窓幅（Noneなら点数の約1/20、7以上）。
            ax: 系列数個のAxesの並び。
        """
        series_map = _as_series_mapping(residuals)
        # 1系列なら、時刻・区間IDは配列のままでもその系列のものとして扱う
        default = next(iter(series_map)) if len(series_map) == 1 else "残差"
        times = _as_series_mapping(time, default) if time is not None else {}
        segs = _as_series_mapping(segments, default) if segments is not None else {}
        unknown = (set(times) | set(segs)) - set(series_map)
        if unknown:
            raise ValueError(f"time / segments に残差に無い系列があります: {sorted(unknown)}")
        data = {}
        for name, values in series_map.items():
            r = as_1d_float(values)
            t = np.asarray(times[name]) if name in times else np.arange(r.size)
            s = np.asarray(segs[name]) if name in segs else None
            data[name] = (t, r, s)
        return cls(data, rolling_window).plot(ax=ax)

    def plot(self, ax: Any = None) -> Self:
        """系列ごとの残差の折れ線を描く。"""
        ensure_japanese_font()
        names = list(self.series)
        if ax is None:
            fig, axes = new_axes(None, nrows=len(names), figsize=(13, 3.2 * len(names) + 0.6))
            axes = np.asarray(axes, dtype=object).reshape(-1)
        else:
            # 系列数個のAxesの並び（1系列なら1個のAxesでもよい）
            axes = np.asarray(ax, dtype=object).reshape(-1)
            if axes.size != len(names):
                raise ValueError(f"ax は {len(names)} 個必要です（{axes.size} 個が渡されました）")
            fig = axes[0].figure
        for a, name in zip(axes, names, strict=True):
            self._plot_one(a, name, *self.series[name])
        self.figure_, self.ax_ = fig, axes if len(names) > 1 else axes[0]
        return self

    def _plot_one(
        self, a: Any, name: str, t: np.ndarray, r: np.ndarray, s: np.ndarray | None
    ) -> None:
        line, boundaries = _line_values(t, r, s)
        a.plot(t, line, color="#4c72b0", linewidth=0.8, label="残差")
        finite = r[np.isfinite(r)]
        window = self.rolling_window or max(7, finite.size // 20)
        # 移動平均は区切り（foldの境目・時刻の空白）ごとに計算し、区切りをまたいで線を結ばない
        cut = np.isnan(line) & np.isfinite(r)
        rolling = (
            pl.DataFrame({"v": r, "block": np.cumsum(cut)})
            .with_columns(pl.col("v").fill_nan(None))
            .select(pl.col("v").rolling_mean(window, min_samples=max(1, window // 2)).over("block"))
            .to_series()
            .to_numpy()
            .astype(np.float64)
        )
        rolling[cut] = np.nan
        a.plot(t, rolling, color="#c44e52", linewidth=1.5, label=f"移動平均（{window}点）")
        if finite.size > 1:
            sd = float(finite.std(ddof=1))
            a.axhspan(-2 * sd, 2 * sd, color="gray", alpha=0.12, label="±2σ")
        a.axhline(0, color="black", linewidth=0.8)
        for b in boundaries:
            a.axvline(t[b], color="gray", linestyle=":", linewidth=1)
        a.set_title(f"{name}: 残差の推移（n={finite.size:,}）")
        a.set_xlabel("時刻")
        a.set_ylabel("残差（実測値 − 予測値）")
        a.legend(loc="upper left", fontsize=8, ncols=3)


def _format_p(p: Any) -> str:
    """p値の表示（`p=0.123` / `p<0.001`）。"""
    if p is None or not np.isfinite(p):
        return "p=－"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _format_stat(x: Any) -> str:
    if x is None or not np.isfinite(x):
        return "－"
    return f"{x:.3g}"


class TimeSeriesResidualDiagnosticsDisplay:
    """1系列の残差診断を1枚にまとめた図。

    1段目: 残差の推移（`ResidualTimeSeriesDisplay`）、2段目: ヒストグラムと正規Q-Q、
    3段目: ACFとPACF、4段目: 検定（Ljung-Box・Jarque-Bera・ADF・KPSS）の結果の表。

    Attributes:
        tests: `residual_tests` の要約表（1行）。
        ljung_box: Ljung-Box検定の全ラグの結果。
        figure_: 描画したFigure。
        ax_: 描画したAxes（`line`, `hist`, `qq`, `acf`, `pacf`, `table` の辞書）。
    """

    figure_: plt.Figure
    ax_: dict[str, Any]

    def __init__(self, tests: pl.DataFrame, ljung_box: pl.DataFrame) -> None:
        self.tests = tests
        self.ljung_box = ljung_box

    @classmethod
    def from_residuals(
        cls,
        residuals: Any,
        *,
        time: Any = None,
        segments: Any = None,
        nlags: int = 40,
        lags: Sequence[int] | None = None,
        seasonal_period: int | None = None,
        regression: str = "c",
        alpha: float = 0.05,
        title: str = "",
    ) -> Self:
        """時刻順の残差（1系列）から検定・描画をまとめて行う。

        Args:
            residuals: 時刻順の残差。
            time: 各残差の時刻（Noneなら並び順の番号）。
            segments: 各残差の区間ID（foldなど）。
            nlags: ACF/PACFの最大ラグ。
            lags: Ljung-Boxのラグ（Noneなら既定値）。
            seasonal_period: Ljung-Boxの既定のラグに加える季節周期。
            regression: 単位根検定の確定項（`c` / `ct`）。
            alpha: 判定に使う有意水準。
            title: 図全体の見出し。
        """
        ensure_japanese_font()
        r = as_1d_float(residuals)
        summary, lb = residual_tests(
            {"残差": r},
            lags=lags,
            seasonal_period=seasonal_period,
            regression=regression,
            alpha=alpha,
        )
        disp = cls(summary, lb)
        fig, axes = plt.subplot_mosaic(
            [["line", "line"], ["hist", "qq"], ["acf", "pacf"], ["table", "table"]],
            figsize=(13, 15),
            height_ratios=[1.0, 1.1, 0.9, 0.55],
            constrained_layout=True,
        )
        ResidualTimeSeriesDisplay.from_residuals(
            {"残差": r},
            time=None if time is None else {"残差": time},
            segments=None if segments is None else {"残差": segments},
            ax=[axes["line"]],
        )
        finite = r[np.isfinite(r)]
        if finite.size > 1:
            ResidualDistributionDisplay.from_residuals(finite, ax=axes["hist"])
            QQPlotDisplay.from_residuals(finite, ax=axes["qq"])
            # 分布の見出しが長いため、統計量は検定の表に任せて短くする
            axes["hist"].set_title("残差のヒストグラム")
        ResidualCorrelogramDisplay.from_residuals(
            {"残差": r}, nlags=nlags, ax=[[axes["acf"], axes["pacf"]]]
        )
        # 1系列なので、各パネルの見出しから系列名（「残差: 」）を外す
        for key in ("line", "acf", "pacf"):
            axes[key].set_title(axes[key].get_title().removeprefix("残差: "))
        disp._plot_table(axes["table"], alpha)
        fig.suptitle(title or "時系列の残差診断")
        disp.figure_, disp.ax_ = fig, axes
        return disp

    def _plot_table(self, ax: Any, alpha: float) -> None:
        """検定結果の表を描く。"""
        row = self.tests.row(0, named=True)
        lb_text = ", ".join(
            f"lag {k}: {_format_p(p)}"
            for k, p in zip(self.ljung_box["lag"], self.ljung_box["p_value"], strict=True)
        )
        cells = [
            [
                "Ljung-Box",
                "自己相関なし",
                lb_text or "－",
                f"自己相関{row['autocorrelation']}",
            ],
            [
                "Jarque-Bera",
                "正規分布",
                f"統計量={_format_stat(row['jarque_bera_statistic'])}, "
                f"{_format_p(row['jarque_bera_p_value'])}",
                f"正規性を{row['normality']}",
            ],
            [
                "ADF",
                "単位根あり（非定常）",
                f"統計量={_format_stat(row['adf_statistic'])}, {_format_p(row['adf_p_value'])}",
                row["stationarity"],
            ],
            [
                "KPSS",
                "定常",
                f"統計量={_format_stat(row['kpss_statistic'])}, "
                f"{_format_p(row['kpss_p_value'])}（0.01〜0.1で打ち切り）",
                "（ADFと合わせて判定）",
            ],
        ]
        ax.set_axis_off()
        table = ax.table(
            cellText=cells,
            colLabels=["検定", "帰無仮説", "結果", f"判定（有意水準{alpha:g}）"],
            colWidths=[0.12, 0.18, 0.45, 0.25],
            loc="center",
            cellLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
