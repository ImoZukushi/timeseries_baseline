"""time_series_eda モジュールのテスト。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from eda import time_series_eda as tse


def _dates(n: int, start: str = "2020-01-01") -> pl.Series:
    start_date = dt.date.fromisoformat(start)
    return pl.Series([start_date + dt.timedelta(days=i) for i in range(n)])


# --- sanitize_filename_component ----------------------------------------------


def test_sanitize_filename_component_replaces_unsafe_chars() -> None:
    assert tse.sanitize_filename_component('a:b/c\\d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"


def test_sanitize_filename_component_leaves_safe_text_unchanged() -> None:
    assert (
        tse.sanitize_filename_component("wind_0_下条川__風速(瞬時)") == "wind_0_下条川__風速(瞬時)"
    )


# --- find_datetime_column / find_numeric_columns ----------------------------


def test_find_datetime_column_found() -> None:
    df = pl.DataFrame({"ts": _dates(3), "v": [1, 2, 3]})
    assert tse.find_datetime_column(df) == "ts"


def test_find_datetime_column_none() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    assert tse.find_datetime_column(df) is None


def test_find_numeric_columns_excludes_given_column() -> None:
    df = pl.DataFrame({"ts": _dates(3), "v": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})
    assert tse.find_numeric_columns(df, exclude=("ts",)) == ["v"]


# --- find_grouping_column ----------------------------------------------------


def test_find_grouping_column_no_duplicates_returns_none() -> None:
    df = pl.DataFrame({"ts": _dates(5), "v": range(5)})
    assert tse.find_grouping_column(df, "ts") is None


def test_find_grouping_column_resolved_by_single_column() -> None:
    dates = list(_dates(3)) * 2
    df = pl.DataFrame(
        {
            "ts": dates,
            "site": ["A", "A", "A", "B", "B", "B"],
            "v": range(6),
        }
    )
    assert tse.find_grouping_column(df, "ts") == "site"


def test_find_grouping_column_unresolvable_returns_none() -> None:
    # (ts, key1) だけでも (ts, key2) だけでも重複が解消できないケース
    df = pl.DataFrame(
        {
            "ts": ["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01"],
            "key1": ["A", "A", "B", "B"],
            "key2": ["X", "Y", "X", "Y"],
            "v": range(4),
        }
    )
    assert tse.find_grouping_column(df, "ts") is None


# --- to_log_series / build_transformed_series --------------------------------


def test_to_log_series_masks_non_positive_values() -> None:
    values = pl.Series("v", [1.0, np.e, -1.0, 0.0, None])
    result = tse.to_log_series(values)
    assert result.to_list()[0] == pytest.approx(0.0)
    assert result.to_list()[1] == pytest.approx(1.0)
    assert result.to_list()[2] is None
    assert result.to_list()[3] is None
    assert result.to_list()[4] is None
    assert result.name == "v"


def test_build_transformed_series_has_expected_keys() -> None:
    values = pl.Series("v", [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    result = tse.build_transformed_series(values, seasonal_period=2)
    assert set(result.keys()) == set(tse.SERIES_LABELS.keys())


def test_build_transformed_series_diff_values() -> None:
    values = pl.Series("v", [10.0, 12.0, 15.0, 11.0])
    result = tse.build_transformed_series(values, seasonal_period=2)
    assert result["diff"].to_list() == [None, 2.0, 3.0, -4.0]
    assert result["seasonal_diff"].to_list() == [None, None, 5.0, -1.0]


def test_build_transformed_series_log_diff_values() -> None:
    values = pl.Series("v", [1.0, np.e, np.e**2])
    result = tse.build_transformed_series(values, seasonal_period=1)
    log_diff = result["log_diff"].to_list()
    assert log_diff[0] is None
    assert log_diff[1] == pytest.approx(1.0)
    assert log_diff[2] == pytest.approx(1.0)


# --- moving_average -----------------------------------------------------------


def test_moving_average_basic() -> None:
    values = pl.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = tse.moving_average(values, window=3)
    assert result.to_list() == [None, None, 2.0, 3.0, 4.0]


# --- break_line_at_gaps ---------------------------------------------------------


def test_break_line_at_gaps_breaks_large_gap() -> None:
    x = np.array([0, 1, 2, 3, 100, 101, 102], dtype=float)
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    result = tse.break_line_at_gaps(x, y, gap_factor=10.0)
    assert np.isnan(result[4])
    assert not np.isnan(result[:4]).any()
    assert not np.isnan(result[5:]).any()


def test_break_line_at_gaps_no_gap_leaves_unchanged() -> None:
    x = np.arange(10, dtype=float)
    y = np.arange(10, dtype=float)
    result = tse.break_line_at_gaps(x, y, gap_factor=10.0)
    assert not np.isnan(result).any()
    np.testing.assert_array_equal(result, y)


def test_break_line_at_gaps_with_datetime64() -> None:
    x = np.array(
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-06-01", "2020-06-02"],
        dtype="datetime64[D]",
    )
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = tse.break_line_at_gaps(x, y, gap_factor=10.0)
    assert np.isnan(result[3])
    assert not np.isnan(result[:3]).any()
    assert not np.isnan(result[4]).any()


def test_break_line_at_gaps_too_short_returns_unchanged() -> None:
    x = np.array([0.0, 1.0])
    y = np.array([1.0, 2.0])
    result = tse.break_line_at_gaps(x, y)
    np.testing.assert_array_equal(result, y)


# --- compute_acf / compute_pacf ------------------------------------------------


def test_compute_acf_none_when_insufficient_data() -> None:
    values = pl.Series([1.0, 2.0, 3.0])
    assert tse.compute_acf(values, nlags=5) is None


def test_compute_acf_lag_zero_is_one() -> None:
    rng = np.random.default_rng(0)
    values = pl.Series(rng.normal(size=200))
    result = tse.compute_acf(values, nlags=10)
    assert result is not None
    assert result[0] == pytest.approx(1.0)
    assert len(result) == 11


def test_compute_acf_truncates_to_max_points() -> None:
    rng = np.random.default_rng(0)
    values = pl.Series(rng.normal(size=100_000))
    result = tse.compute_acf(values, nlags=5, max_points=1_000)
    assert result is not None
    assert result[0] == pytest.approx(1.0)


def test_compute_pacf_none_when_insufficient_data() -> None:
    values = pl.Series([1.0, 2.0, 3.0])
    assert tse.compute_pacf(values, nlags=5) is None


def test_compute_pacf_lag_zero_is_one() -> None:
    rng = np.random.default_rng(0)
    values = pl.Series(rng.normal(size=200))
    result = tse.compute_pacf(values, nlags=10)
    assert result is not None
    assert result[0] == pytest.approx(1.0)


def test_compute_acf_ignores_nulls() -> None:
    rng = np.random.default_rng(1)
    values = pl.Series([None, None, *rng.normal(size=200).tolist()])
    result = tse.compute_acf(values, nlags=5)
    assert result is not None
    assert result[0] == pytest.approx(1.0)


# --- plotting -------------------------------------------------------------


def _sample_series_dict(
    n: int = 200, seasonal_period: int = 7
) -> tuple[pl.Series, dict[str, pl.Series]]:
    dt_series = pl.Series([dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)])
    rng = np.random.default_rng(0)
    values = pl.Series("v", np.abs(rng.normal(loc=10, scale=2, size=n)))
    return dt_series, tse.build_transformed_series(values, seasonal_period=seasonal_period)


def test_plot_series_with_moving_average_creates_file(tmp_path: Path) -> None:
    dt_series, series_dict = _sample_series_dict()
    out_path = tmp_path / "ma.png"
    created = tse.plot_series_with_moving_average(dt_series, series_dict, "sample", "v", out_path)
    assert created is True
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_series_with_moving_average_false_when_empty(tmp_path: Path) -> None:
    dt_series = pl.Series([dt.date(2020, 1, 1)])
    empty = pl.Series("v", [None], dtype=pl.Float64)
    series_dict = tse.build_transformed_series(empty)
    created = tse.plot_series_with_moving_average(
        dt_series, series_dict, "sample", "v", tmp_path / "out.png"
    )
    assert created is False
    assert not (tmp_path / "out.png").exists()


def test_plot_acf_correlogram_creates_file(tmp_path: Path) -> None:
    _, series_dict = _sample_series_dict()
    out_path = tmp_path / "acf.png"
    created = tse.plot_acf_correlogram(series_dict, "sample", "v", out_path)
    assert created is True
    assert out_path.exists()


def test_plot_pacf_correlogram_creates_file(tmp_path: Path) -> None:
    _, series_dict = _sample_series_dict()
    out_path = tmp_path / "pacf.png"
    created = tse.plot_pacf_correlogram(series_dict, "sample", "v", out_path)
    assert created is True
    assert out_path.exists()


# --- run_time_series_checks (end-to-end) --------------------------------------


def test_run_time_series_checks_returns_empty_without_datetime_column(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    result = tse.run_time_series_checks(df, "sample", tmp_path)
    assert result == []
    assert list(tmp_path.glob("*.png")) == []


def test_run_time_series_checks_single_series(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200
    df = pl.DataFrame(
        {
            "ts": [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)],
            "value": np.abs(rng.normal(loc=10, scale=2, size=n)),
        }
    )
    result = tse.run_time_series_checks(df, "sample", tmp_path, nlags=10)
    assert result == ["sample:value"]
    assert (tmp_path / "sample__value__series_with_moving_average.png").exists()
    assert (tmp_path / "sample__value__acf_correlogram.png").exists()
    assert (tmp_path / "sample__value__pacf_correlogram.png").exists()


def test_run_time_series_checks_grouped_panel_data(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 60
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    df = pl.concat(
        [
            pl.DataFrame(
                {
                    "ts": dates,
                    "site": [site] * n,
                    "value": np.abs(rng.normal(loc=10, scale=2, size=n)),
                }
            )
            for site in ["A", "B"]
        ]
    )
    result = tse.run_time_series_checks(df, "sample", tmp_path, nlags=5)
    assert sorted(result) == ["sample_A:value", "sample_B:value"]
    assert (tmp_path / "sample_A__value__series_with_moving_average.png").exists()
    assert (tmp_path / "sample_B__value__acf_correlogram.png").exists()


def test_run_time_series_checks_sanitizes_group_value_in_filename(tmp_path: Path) -> None:
    # グループ列の値に ":" が含まれるケース（例: 時刻文字列）でファイル名が壊れないことを確認する。
    rng = np.random.default_rng(0)
    n = 30
    dates = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    df = pl.concat(
        [
            pl.DataFrame(
                {
                    "ts": dates,
                    "time_of_day": [time_str] * n,
                    "value": np.abs(rng.normal(loc=10, scale=2, size=n)),
                }
            )
            for time_str in ["06:41:00", "19:33:00"]
        ]
    )
    result = tse.run_time_series_checks(df, "sample", tmp_path, nlags=5)
    assert sorted(result) == ["sample_06:41:00:value", "sample_19:33:00:value"]
    assert (tmp_path / "sample_06_41_00__value__series_with_moving_average.png").exists()
    assert (tmp_path / "sample_19_33_00__value__acf_correlogram.png").exists()
    assert not any(":" in p.name for p in tmp_path.glob("*.png"))


def test_run_time_series_checks_unresolvable_duplicates_returns_empty(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "ts": ["2020-01-01", "2020-01-01", "2020-01-01", "2020-01-01"],
            "key1": ["A", "A", "B", "B"],
            "key2": ["X", "Y", "X", "Y"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    ).with_columns(pl.col("ts").str.to_datetime())
    result = tse.run_time_series_checks(df, "sample", tmp_path)
    assert result == []
    assert list(tmp_path.glob("*.png")) == []
