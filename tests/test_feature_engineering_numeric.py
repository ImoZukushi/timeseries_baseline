"""feature_engineering.numeric のテスト。"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from feature_engineering.numeric import (
    FixedPowerTransformer,
    LogTransformer,
    PolarsPowerTransformer,
    ReciprocalTransformer,
    SqrtTransformer,
)

# --- LogTransformer -------------------------------------------------------------


def test_log_transformer_natural_log() -> None:
    train = pl.DataFrame({"x": [1.0, math.e, math.e**2]})
    t = LogTransformer("x").fit(train)
    out = t.transform(train)
    assert out["x"].to_list() == pytest.approx([0.0, 1.0, 2.0])


def test_log_transformer_custom_base() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 8.0]})
    t = LogTransformer("x", base=2.0).fit(train)
    out = t.transform(train)
    assert out["x"].to_list() == pytest.approx([0.0, 1.0, 3.0])


def test_log_transformer_non_positive_becomes_null() -> None:
    train = pl.DataFrame({"x": [1.0, 0.0, -1.0]})
    t = LogTransformer("x").fit(train)
    out = t.transform(train)
    assert out["x"].to_list()[1] is None
    assert out["x"].to_list()[2] is None


def test_log_transformer_round_trip() -> None:
    train = pl.DataFrame({"x": [1.0, 2.5, 10.0, 100.0]})
    t = LogTransformer("x", base=10.0).fit(train)
    back = t.inverse_transform(t.transform(train))
    assert back["x"].to_list() == pytest.approx(train["x"].to_list())


def test_log_transformer_leak_invariance() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0]})
    test = pl.DataFrame({"x": [5.0, 50.0]})
    t = LogTransformer("x").fit(train)
    alone = t.transform(test)["x"].to_list()
    combined = t.transform(pl.concat([train, test]))["x"].to_list()[-2:]
    assert alone == pytest.approx(combined)


# --- ReciprocalTransformer -------------------------------------------------------


def test_reciprocal_transformer_basic() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, -4.0]})
    t = ReciprocalTransformer("x").fit(train)
    out = t.transform(train)
    assert out["x"].to_list() == pytest.approx([1.0, 0.5, -0.25])


def test_reciprocal_transformer_zero_becomes_null() -> None:
    train = pl.DataFrame({"x": [1.0, 0.0]})
    t = ReciprocalTransformer("x").fit(train)
    out = t.transform(train)
    assert out["x"].to_list()[1] is None


def test_reciprocal_transformer_round_trip() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, -4.0, 0.25]})
    t = ReciprocalTransformer("x").fit(train)
    back = t.inverse_transform(t.transform(train))
    assert back["x"].to_list() == pytest.approx(train["x"].to_list())


# --- SqrtTransformer --------------------------------------------------------------


def test_sqrt_transformer_basic() -> None:
    train = pl.DataFrame({"x": [4.0, 9.0, 0.0]})
    t = SqrtTransformer("x").fit(train)
    out = t.transform(train)
    assert out["x"].to_list() == pytest.approx([2.0, 3.0, 0.0])


def test_sqrt_transformer_negative_becomes_null() -> None:
    train = pl.DataFrame({"x": [4.0, -1.0]})
    t = SqrtTransformer("x").fit(train)
    out = t.transform(train)
    assert out["x"].to_list()[1] is None


def test_sqrt_transformer_round_trip() -> None:
    train = pl.DataFrame({"x": [4.0, 9.0, 100.0]})
    t = SqrtTransformer("x").fit(train)
    back = t.inverse_transform(t.transform(train))
    assert back["x"].to_list() == pytest.approx(train["x"].to_list())


# --- FixedPowerTransformer ------------------------------------------------------


def test_fixed_power_transformer_basic() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    t = FixedPowerTransformer("x", power=3).fit(train)
    out = t.transform(train)
    assert out["x"].to_list() == pytest.approx([1.0, 8.0, 27.0])


def test_fixed_power_transformer_round_trip() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    t = FixedPowerTransformer("x", power=3).fit(train)
    back = t.inverse_transform(t.transform(train))
    assert back["x"].to_list() == pytest.approx(train["x"].to_list())


# --- PolarsPowerTransformer (Box-Cox / Yeo-Johnson) ------------------------------


def test_box_cox_round_trip() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 100.0, 5.0, 7.0]})
    t = PolarsPowerTransformer("x", method="box-cox").fit(train)
    back = t.inverse_transform(t.transform(train))
    assert back["x"].to_list() == pytest.approx(train["x"].to_list())


def test_box_cox_rejects_non_positive_training_data() -> None:
    train = pl.DataFrame({"x": [1.0, 0.0, 2.0]})
    with pytest.raises(ValueError):
        PolarsPowerTransformer("x", method="box-cox").fit(train)


def test_yeo_johnson_handles_non_positive_values() -> None:
    train = pl.DataFrame({"x": [-5.0, 0.0, 1.0, 2.0, 100.0]})
    t = PolarsPowerTransformer("x", method="yeo-johnson").fit(train)
    out = t.transform(train)
    assert out["x"].null_count() == 0


def test_yeo_johnson_round_trip() -> None:
    train = pl.DataFrame({"x": [-5.0, 0.0, 1.0, 2.0, 100.0]})
    t = PolarsPowerTransformer("x", method="yeo-johnson").fit(train)
    back = t.inverse_transform(t.transform(train))
    assert back["x"].to_list() == pytest.approx(train["x"].to_list(), abs=1e-6)


def test_power_transformer_leak_invariance() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    test = pl.DataFrame({"x": [10.0, 20.0]})
    t = PolarsPowerTransformer("x", method="yeo-johnson").fit(train)
    alone = t.transform(test)["x"].to_list()
    combined = t.transform(pl.concat([train, test]))["x"].to_list()[-2:]
    assert alone == pytest.approx(combined)


def test_power_transformer_does_not_standardize() -> None:
    # standardize=Falseを想定しているため、変換結果の平均が必ずしも0にならないことを確認
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})
    t = PolarsPowerTransformer("x", method="box-cox").fit(train)
    out = t.transform(train)["x"].to_numpy()
    assert not np.isclose(out.mean(), 0.0, atol=1e-2)
