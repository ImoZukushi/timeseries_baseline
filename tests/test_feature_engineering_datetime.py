"""feature_engineering.datetime_features のテスト。"""

from __future__ import annotations

import datetime as dt
import math

import polars as pl
import pytest

from feature_engineering.datetime_features import (
    CyclicalFeaturesEncoder,
    DatetimeFeaturesExtractor,
    ElapsedTimeTransformer,
)

# --- DatetimeFeaturesExtractor ----------------------------------------------------


def test_datetime_features_extractor_all_components() -> None:
    train = pl.DataFrame({"ts": [dt.datetime(2024, 3, 4, 13, 45, 30)]})
    ext = DatetimeFeaturesExtractor("ts").fit(train)
    out = ext.transform(train)
    row = out.row(0, named=True)
    assert row["ts_year"] == 2024
    assert row["ts_month"] == 3
    assert row["ts_day"] == 4
    assert row["ts_weekday"] == 1  # 2024-03-04は月曜日
    assert row["ts_hour"] == 13
    assert row["ts_minute"] == 45
    assert row["ts_second"] == 30


def test_datetime_features_extractor_selected_components_only() -> None:
    train = pl.DataFrame({"ts": [dt.datetime(2024, 3, 4)]})
    ext = DatetimeFeaturesExtractor("ts", components=("year", "month")).fit(train)
    out = ext.transform(train)
    assert set(out.columns) == {"ts", "ts_year", "ts_month"}


def test_datetime_features_extractor_preserves_original_column() -> None:
    train = pl.DataFrame({"ts": [dt.datetime(2024, 3, 4)]})
    ext = DatetimeFeaturesExtractor("ts", components=("year",)).fit(train)
    out = ext.transform(train)
    assert out["ts"].to_list() == train["ts"].to_list()


def test_datetime_features_extractor_rejects_unknown_component() -> None:
    train = pl.DataFrame({"ts": [dt.datetime(2024, 3, 4)]})
    with pytest.raises(ValueError):
        DatetimeFeaturesExtractor("ts", components=("century",)).fit(train)


# --- ElapsedTimeTransformer --------------------------------------------------------


def test_elapsed_time_from_train_minimum_in_days() -> None:
    train = pl.DataFrame({"ts": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 5)]})
    et = ElapsedTimeTransformer("ts", unit="days").fit(train)
    out = et.transform(train)
    assert out["ts_elapsed_days"].to_list() == pytest.approx([0.0, 4.0])


def test_elapsed_time_test_data_can_precede_reference() -> None:
    train = pl.DataFrame({"ts": [dt.datetime(2024, 1, 10)]})
    test = pl.DataFrame({"ts": [dt.datetime(2024, 1, 5)]})
    et = ElapsedTimeTransformer("ts", unit="days").fit(train)
    out = et.transform(test)
    assert out["ts_elapsed_days"].to_list() == pytest.approx([-5.0])


def test_elapsed_time_units() -> None:
    train = pl.DataFrame({"ts": [dt.datetime(2024, 1, 1, 0, 0, 0)]})
    test = pl.DataFrame({"ts": [dt.datetime(2024, 1, 1, 2, 0, 0)]})
    for unit, expected in [("seconds", 7200.0), ("minutes", 120.0), ("hours", 2.0)]:
        et = ElapsedTimeTransformer("ts", unit=unit).fit(train)
        out = et.transform(test)
        assert out[f"ts_elapsed_{unit}"].to_list() == pytest.approx([expected])


def test_elapsed_time_leak_invariance() -> None:
    train = pl.DataFrame({"ts": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 3)]})
    test = pl.DataFrame({"ts": [dt.datetime(2024, 1, 10)]})
    et = ElapsedTimeTransformer("ts").fit(train)
    alone = et.transform(test)["ts_elapsed_days"].to_list()
    combined = et.transform(pl.concat([train, test]))["ts_elapsed_days"].to_list()[-1:]
    assert alone == pytest.approx(combined)


# --- CyclicalFeaturesEncoder --------------------------------------------------------


def test_cyclical_features_boundary_continuity() -> None:
    # 周期の始点(0)と終点(period)は同じ角度になるはず
    train = pl.DataFrame({"m": [0, 12]})
    enc = CyclicalFeaturesEncoder("m", period=12).fit(train)
    out = enc.transform(train)
    assert out["m_sin"].to_list() == pytest.approx([0.0, 0.0], abs=1e-9)
    assert out["m_cos"].to_list() == pytest.approx([1.0, 1.0], abs=1e-9)


def test_cyclical_features_quarter_period() -> None:
    train = pl.DataFrame({"h": [0, 6, 12, 18]})
    enc = CyclicalFeaturesEncoder("h", period=24).fit(train)
    out = enc.transform(train)
    assert out["h_sin"].to_list() == pytest.approx([0.0, 1.0, 0.0, -1.0], abs=1e-9)
    assert out["h_cos"].to_list() == pytest.approx([1.0, 0.0, -1.0, 0.0], abs=1e-9)


def test_cyclical_features_distance_is_symmetric_across_boundary() -> None:
    # 11(12ヶ月周期で0の直前)と0は隣接しているのでsin/cos平面上でも近く、
    # 11と5（周期の反対側、最も離れている）は遠いはず
    train = pl.DataFrame({"m": [11, 0, 5]})
    enc = CyclicalFeaturesEncoder("m", period=12).fit(train)
    out = enc.transform(train)

    def distance(i: int, j: int) -> float:
        dx = out["m_sin"][i] - out["m_sin"][j]
        dy = out["m_cos"][i] - out["m_cos"][j]
        return math.hypot(dx, dy)

    adjacent = distance(0, 1)  # 11 と 0（1ヶ月差）
    opposite = distance(0, 2)  # 11 と 5（6ヶ月差、周期上最も遠い）
    assert adjacent < opposite
