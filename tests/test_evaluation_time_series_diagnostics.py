"""evaluation.time_series_diagnostics と、残差の Display の from_residuals のテスト。"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from evaluation.residuals import QQPlotDisplay, ResidualDistributionDisplay
from evaluation.time_series_diagnostics import (
    NON_STATIONARY,
    NOT_COMPUTED,
    STATIONARY,
    ResidualTimeSeriesDisplay,
    TimeSeriesResidualDiagnosticsDisplay,
    default_ljung_box_lags,
    jarque_bera_test,
    ljung_box_test,
    residual_tests,
    unit_root_test,
)

N = 400


def _white_noise(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=N)


def _ar1(phi: float = 0.8) -> np.ndarray:
    eps = _white_noise(1)
    x = np.zeros(N)
    for i in range(1, N):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def _random_walk() -> np.ndarray:
    return np.cumsum(_white_noise(2))


# --- 検定 -----------------------------------------------------------------------------------


def test_ljung_box_detects_autocorrelation() -> None:
    assert ljung_box_test(_white_noise())["p_value"][0] > 0.05
    assert ljung_box_test(_ar1())["p_value"][0] < 0.01


def test_ljung_box_lags() -> None:
    assert default_ljung_box_lags(400) == [10]
    assert default_ljung_box_lags(30) == [6]
    # 季節周期は点数の半分未満なら加え、長すぎれば加えない
    assert default_ljung_box_lags(400, seasonal_period=30) == [10, 30]
    assert default_ljung_box_lags(400, seasonal_period=365) == [10]
    # 点数の半分以上のラグは除く
    assert ljung_box_test(_white_noise(), lags=[5, 300])["lag"].to_list() == [5]


def test_jarque_bera_detects_non_normality() -> None:
    assert jarque_bera_test(_white_noise())["p_value"] > 0.05
    rng = np.random.default_rng(3)
    assert jarque_bera_test(rng.exponential(size=N))["p_value"] < 0.01
    assert jarque_bera_test(rng.standard_t(3, size=N))["p_value"] < 0.01


def test_unit_root_conclusions() -> None:
    assert unit_root_test(_white_noise())["conclusion"] == STATIONARY
    assert unit_root_test(_random_walk())["conclusion"] == NON_STATIONARY


@pytest.mark.parametrize(
    "values",
    [np.arange(5, dtype=float), np.ones(100), np.full(100, np.nan)],
    ids=["short", "constant", "all_nan"],
)
def test_tests_return_nan_when_not_computable(values: np.ndarray) -> None:
    assert ljung_box_test(values).height == 0
    assert np.isnan(jarque_bera_test(values)["p_value"])
    assert unit_root_test(values)["conclusion"] == NOT_COMPUTED


def test_tests_ignore_missing_values() -> None:
    values = _white_noise()
    values[::10] = np.nan
    assert ljung_box_test(values).height == 1
    assert np.isfinite(jarque_bera_test(values)["p_value"])


def test_residual_tests_one_row_per_series() -> None:
    summary, lb = residual_tests(
        {"wn": _white_noise(), "ar": _ar1(), "short": np.arange(3.0)}, seasonal_period=30
    )
    assert summary["series"].to_list() == ["wn", "ar", "short"]
    rows = {r["series"]: r for r in summary.iter_rows(named=True)}
    assert rows["wn"]["autocorrelation"] == "なし"
    assert rows["ar"]["autocorrelation"] == "あり"
    assert rows["short"]["autocorrelation"] == NOT_COMPUTED
    # 要約表には最大のラグの結果を載せ、Ljung-Box表には全ラグを載せる
    assert rows["wn"]["ljung_box_lag"] == 30
    assert lb.filter(lb["series"] == "wn")["lag"].to_list() == [10, 30]
    assert "short" not in lb["series"].to_list()


# --- 図 -------------------------------------------------------------------------------------


def test_timeseries_display_breaks_line_at_segments() -> None:
    values = _white_noise()
    segments = np.repeat([0, 1], N // 2)
    disp = ResidualTimeSeriesDisplay.from_residuals({"A": values}, segments=segments)
    line = disp.ax_.lines[0].get_ydata()
    # 区間が変わる最初の点で線が切れる
    assert np.isnan(line[N // 2])
    assert np.isfinite(line[N // 2 - 1])
    # 移動平均も区間の境目で切れ、境目の前後は各区間の値だけで計算する
    rolling = disp.ax_.lines[1].get_ydata()
    assert np.isnan(rolling[N // 2])
    assert rolling[N // 2 + 10] == pytest.approx(values[N // 2 : N // 2 + 11].mean())
    plt.close(disp.figure_)


def test_timeseries_display_multiple_series_and_time_gaps() -> None:
    time = np.concatenate([np.arange(100), np.arange(1000, 1100)])
    values = {"A": _white_noise()[:200], "B": _ar1()[:200]}
    disp = ResidualTimeSeriesDisplay.from_residuals(values, time={"A": time, "B": time})
    assert disp.ax_.shape == (2,)
    # 時刻の大きな空白でも線が切れる
    assert np.isnan(disp.ax_[0].lines[0].get_ydata()[100])
    plt.close(disp.figure_)


def test_diagnostics_display_layout() -> None:
    disp = TimeSeriesResidualDiagnosticsDisplay.from_residuals(_ar1(), seasonal_period=30)
    assert set(disp.ax_) == {"line", "hist", "qq", "acf", "pacf", "table"}
    assert disp.tests.height == 1
    assert disp.tests["autocorrelation"][0] == "あり"
    assert disp.ljung_box["lag"].to_list() == [10, 30]
    assert disp.ax_["table"].tables
    plt.close(disp.figure_)


def test_from_residuals_matches_from_predictions() -> None:
    y_true = _white_noise() + 5
    y_pred = np.full(N, 5.2)
    a = QQPlotDisplay.from_predictions(y_true, y_pred)
    b = QQPlotDisplay.from_residuals(y_true - y_pred)
    assert a.r == pytest.approx(b.r)
    c = ResidualDistributionDisplay.from_predictions(y_true, y_pred)
    d = ResidualDistributionDisplay.from_residuals(y_true - y_pred)
    assert c.summary == pytest.approx(d.summary)
    for disp in (a, b, c, d):
        plt.close(disp.figure_)
