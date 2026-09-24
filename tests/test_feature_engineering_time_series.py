"""feature_engineering.time_series のテスト。"""

from __future__ import annotations

import polars as pl
import pytest

from feature_engineering.time_series import (
    LagFeatureGenerator,
    MovingAverageTransformer,
    RateOfChangeTransformer,
)

# --- LagFeatureGenerator -----------------------------------------------------------


def test_lag_feature_generator_basic() -> None:
    train = pl.DataFrame({"x": [10.0, 20.0, 30.0, 40.0]})
    gen = LagFeatureGenerator("x", lags=[1, 2]).fit(train)
    out = gen.transform(train)
    assert out["x_lag_1"].to_list() == [None, 10.0, 20.0, 30.0]
    assert out["x_lag_2"].to_list() == [None, None, 10.0, 20.0]


def test_lag_feature_generator_preserves_original_column() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0]})
    gen = LagFeatureGenerator("x", lags=[1]).fit(train)
    out = gen.transform(train)
    assert out["x"].to_list() == [1.0, 2.0]


def test_lag_feature_generator_rejects_non_positive_lags() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError):
        LagFeatureGenerator("x", lags=[1, 0]).fit(train)
    with pytest.raises(ValueError):
        LagFeatureGenerator("x", lags=[-1]).fit(train)


def test_lag_feature_generator_rejects_empty_lags() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError):
        LagFeatureGenerator("x", lags=[]).fit(train)


def test_lag_feature_generator_window_larger_than_series_is_all_null() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0]})
    gen = LagFeatureGenerator("x", lags=[10]).fit(train)
    out = gen.transform(train)
    assert out["x_lag_10"].to_list() == [None, None]


# --- MovingAverageTransformer -------------------------------------------------------


def test_moving_average_basic() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    ma = MovingAverageTransformer("x", window=3).fit(train)
    out = ma.transform(train)
    assert out["x_ma_3"].to_list() == pytest.approx([None, None, 2.0, 3.0, 4.0], nan_ok=False)


def test_moving_average_min_periods_allows_partial_window() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    ma = MovingAverageTransformer("x", window=3, min_periods=1).fit(train)
    out = ma.transform(train)
    assert out["x_ma_3"].to_list() == pytest.approx([1.0, 1.5, 2.0])


def test_moving_average_is_backward_looking_only() -> None:
    # window=2の移動平均は「自分と直前の値」の平均であり、未来の値を含まない
    train = pl.DataFrame({"x": [0.0, 0.0, 100.0]})
    ma = MovingAverageTransformer("x", window=2).fit(train)
    out = ma.transform(train)
    # 3行目(未来の100)を含む前の行が100の影響を受けていないことを確認
    assert out["x_ma_2"].to_list()[1] == pytest.approx(0.0)


def test_moving_average_window_larger_than_series_is_all_null() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0]})
    ma = MovingAverageTransformer("x", window=5).fit(train)
    out = ma.transform(train)
    assert out["x_ma_5"].to_list() == [None, None]


# --- RateOfChangeTransformer ---------------------------------------------------------


def test_rate_of_change_basic() -> None:
    train = pl.DataFrame({"x": [10.0, 20.0, 15.0]})
    roc = RateOfChangeTransformer("x", periods=1).fit(train)
    out = roc.transform(train)
    assert out["x_roc_1"].to_list() == pytest.approx([None, 1.0, -0.25], nan_ok=False)


def test_rate_of_change_zero_denominator_is_null_not_inf() -> None:
    train = pl.DataFrame({"x": [0.0, 5.0]})
    roc = RateOfChangeTransformer("x", periods=1).fit(train)
    out = roc.transform(train)
    assert out["x_roc_1"].to_list()[1] is None


def test_rate_of_change_round_trip_without_zero_crossing() -> None:
    train = pl.DataFrame({"x": [10.0, 20.0, 15.0, 30.0, 45.0]})
    roc = RateOfChangeTransformer("x", periods=1).fit(train)
    out = roc.transform(train)
    back = roc.inverse_transform(out, initial_value=10.0)
    assert back["x"].to_list() == pytest.approx(train["x"].to_list())


def test_rate_of_change_round_trip_documented_limitation_at_zero() -> None:
    # 値が0を経由すると変化率がnullになり、それ以降は「変化なし」とみなして
    # 復元するため、元の値とは一致しなくなる（原理的な限界としてテストする）。
    train = pl.DataFrame({"x": [10.0, 0.0, 40.0]})
    roc = RateOfChangeTransformer("x", periods=1).fit(train)
    out = roc.transform(train)
    back = roc.inverse_transform(out, initial_value=10.0)
    assert back["x"].to_list() != pytest.approx(train["x"].to_list())
    # 0の直後は「変化なし」として0のまま復元される（有限の比で表現できないため）
    assert back["x"].to_list()[2] == pytest.approx(0.0)


def test_rate_of_change_inverse_transform_rejects_periods_other_than_one() -> None:
    train = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    roc = RateOfChangeTransformer("x", periods=2).fit(train)
    out = roc.transform(train)
    with pytest.raises(NotImplementedError):
        roc.inverse_transform(out, initial_value=1.0)


def test_rate_of_change_leak_invariance() -> None:
    train = pl.DataFrame({"x": [10.0, 20.0, 30.0]})
    test = pl.DataFrame({"x": [1.0, 2.0]})
    roc = RateOfChangeTransformer("x", periods=1).fit(train)
    alone = roc.transform(test)["x_roc_1"].to_list()
    combined = roc.transform(pl.concat([train, test]))["x_roc_1"].to_list()[-2:]
    assert alone[1:] == pytest.approx(combined[1:])


# --- group_by（複数系列） ------------------------------------------------------------


def _panel() -> pl.DataFrame:
    # 系列A・Bが交互に並んだパネルデータ（各系列内は時刻順）
    return pl.DataFrame(
        {
            "s": ["A", "B", "A", "B", "A", "B"],
            "x": [1.0, 100.0, 2.0, 200.0, 4.0, 400.0],
        }
    )


def test_lag_group_by_does_not_mix_series() -> None:
    out = LagFeatureGenerator("x", lags=[1], group_by="s").fit_transform(_panel())
    assert out["x_lag_1"].to_list() == [None, None, 1.0, 100.0, 2.0, 200.0]


def test_lag_without_group_by_keeps_previous_behavior() -> None:
    out = LagFeatureGenerator("x", lags=[1]).fit_transform(_panel())
    assert out["x_lag_1"].to_list() == [None, 1.0, 100.0, 2.0, 200.0, 4.0]


def test_moving_average_group_by() -> None:
    out = MovingAverageTransformer("x", window=2, group_by="s").fit_transform(_panel())
    assert out["x_ma_2"].to_list() == pytest.approx([None, None, 1.5, 150.0, 3.0, 300.0])


def test_rate_of_change_group_by() -> None:
    out = RateOfChangeTransformer("x", periods=1, group_by="s").fit_transform(_panel())
    assert out["x_roc_1"].to_list() == pytest.approx([None, None, 1.0, 1.0, 1.0, 1.0])
