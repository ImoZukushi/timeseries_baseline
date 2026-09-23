"""feature_engineering.scaling のテスト。"""

from __future__ import annotations

import polars as pl
import pytest

from feature_engineering.scaling import (
    MeanNormalizationScaler,
    PolarsMaxAbsScaler,
    PolarsMinMaxScaler,
    PolarsRobustScaler,
)

# --- PolarsMinMaxScaler -----------------------------------------------------------


def test_min_max_scaler_scales_to_unit_interval() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = PolarsMinMaxScaler("x").fit(train)
    out = scaler.transform(train)
    assert out["x"].to_list() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_min_max_scaler_test_values_can_exceed_unit_interval() -> None:
    train = pl.DataFrame({"x": [1.0, 5.0]})
    test = pl.DataFrame({"x": [0.0, 10.0]})
    scaler = PolarsMinMaxScaler("x").fit(train)
    out = scaler.transform(test)
    assert out["x"].to_list() == pytest.approx([-0.25, 2.25])


def test_min_max_scaler_constant_column_no_division_by_zero() -> None:
    train = pl.DataFrame({"x": [5.0, 5.0, 5.0]})
    scaler = PolarsMinMaxScaler("x").fit(train)
    out = scaler.transform(train)
    assert out["x"].to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_min_max_scaler_round_trip() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = PolarsMinMaxScaler("x").fit(train)
    back = scaler.inverse_transform(scaler.transform(train))
    assert back["x"].to_list() == pytest.approx(train["x"].to_list())


def test_min_max_scaler_leak_invariance() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    test = pl.DataFrame({"x": [10.0, -10.0]})
    scaler = PolarsMinMaxScaler("x").fit(train)
    alone = scaler.transform(test)["x"].to_list()
    combined = scaler.transform(pl.concat([train, test]))["x"].to_list()[-2:]
    assert alone == pytest.approx(combined)


# --- PolarsRobustScaler ------------------------------------------------------------


def test_robust_scaler_uses_median_and_iqr() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = PolarsRobustScaler("x").fit(train)
    out = scaler.transform(train)
    # median=3.0, IQR(25-75%)=2.0 -> (x-3)/2
    assert out["x"].to_list() == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0])


def test_robust_scaler_is_not_affected_by_outlier() -> None:
    train_with_outlier = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 1000.0]})
    scaler = PolarsRobustScaler("x").fit(train_with_outlier)
    out = scaler.transform(pl.DataFrame({"x": [3.0]}))
    # 外れ値1000があっても中央値(3.5)付近はスケール後もほぼ0に近い
    assert abs(out["x"][0]) < 1.0


def test_robust_scaler_round_trip() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = PolarsRobustScaler("x").fit(train)
    back = scaler.inverse_transform(scaler.transform(train))
    assert back["x"].to_list() == pytest.approx(train["x"].to_list())


def test_robust_scaler_constant_column_no_division_by_zero() -> None:
    train = pl.DataFrame({"x": [7.0, 7.0, 7.0]})
    scaler = PolarsRobustScaler("x").fit(train)
    out = scaler.transform(train)
    assert out["x"].to_list() == pytest.approx([0.0, 0.0, 0.0])


# --- MeanNormalizationScaler --------------------------------------------------------


def test_mean_normalization_formula() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = MeanNormalizationScaler("x").fit(train)
    out = scaler.transform(train)
    # mean=3.0, range=4.0 -> (x-3)/4
    assert out["x"].to_list() == pytest.approx([-0.5, -0.25, 0.0, 0.25, 0.5])


def test_mean_normalization_constant_column_no_division_by_zero() -> None:
    train = pl.DataFrame({"x": [5.0, 5.0, 5.0]})
    scaler = MeanNormalizationScaler("x").fit(train)
    out = scaler.transform(train)
    assert out["x"].to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_mean_normalization_round_trip() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    scaler = MeanNormalizationScaler("x").fit(train)
    back = scaler.inverse_transform(scaler.transform(train))
    assert back["x"].to_list() == pytest.approx(train["x"].to_list())


def test_mean_normalization_leak_invariance() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    test = pl.DataFrame({"x": [100.0]})
    scaler = MeanNormalizationScaler("x").fit(train)
    alone = scaler.transform(test)["x"].to_list()
    combined = scaler.transform(pl.concat([train, test]))["x"].to_list()[-1:]
    assert alone == pytest.approx(combined)


# --- PolarsMaxAbsScaler -------------------------------------------------------------


def test_max_abs_scaler_scales_by_max_absolute_value() -> None:
    train = pl.DataFrame({"x": [-4.0, 2.0, 4.0]})
    scaler = PolarsMaxAbsScaler("x").fit(train)
    out = scaler.transform(train)
    assert out["x"].to_list() == pytest.approx([-1.0, 0.5, 1.0])


def test_max_abs_scaler_all_zero_column_no_division_by_zero() -> None:
    train = pl.DataFrame({"x": [0.0, 0.0, 0.0]})
    scaler = PolarsMaxAbsScaler("x").fit(train)
    out = scaler.transform(train)
    assert out["x"].to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_max_abs_scaler_round_trip() -> None:
    train = pl.DataFrame({"x": [-4.0, 2.0, 4.0, -1.0]})
    scaler = PolarsMaxAbsScaler("x").fit(train)
    back = scaler.inverse_transform(scaler.transform(train))
    assert back["x"].to_list() == pytest.approx(train["x"].to_list())
