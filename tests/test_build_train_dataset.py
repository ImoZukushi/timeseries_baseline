"""scripts/build_train_dataset.py のテスト。"""

from __future__ import annotations

import build_train_dataset as btd
import polars as pl
import pytest

# --- build_departure_time_table ---------------------------------------------


def test_build_departure_time_table_converts_wide_to_long() -> None:
    diagram_df = pl.DataFrame(
        {
            "停車場名": ["富山", "糸魚川"],
            "3500E": ["6:19", "6:50"],
            "552E": ["6:37", "↓"],
        }
    )

    result = btd.build_departure_time_table(diagram_df)

    assert set(result.columns) == {"停車駅名", "列車番号", "発車時刻"}
    assert result.height == 4  # 2駅 x 2列車番号


def test_build_departure_time_table_arrow_becomes_null() -> None:
    diagram_df = pl.DataFrame(
        {
            "停車場名": ["富山", "糸魚川"],
            "552E": ["6:37", "↓"],
        }
    )

    result = btd.build_departure_time_table(diagram_df)
    row = result.filter((pl.col("停車駅名") == "糸魚川") & (pl.col("列車番号") == "552E"))

    assert row["発車時刻"][0] is None


def test_build_departure_time_table_time_values_are_preserved_as_is() -> None:
    diagram_df = pl.DataFrame(
        {
            "停車場名": ["富山"],
            "3500E": ["6:19"],
        }
    )

    result = btd.build_departure_time_table(diagram_df)

    assert result.row(0, named=True) == {
        "停車駅名": "富山",
        "列車番号": "3500E",
        "発車時刻": "6:19",
    }


# --- join_stop_station_location -----------------------------------------------


def test_join_stop_station_location_adds_location_columns() -> None:
    train_df = pl.DataFrame({"列車番号": ["3500E"], "停車駅名": ["富山"]})
    location_df = pl.DataFrame(
        {
            "停車場名": ["富山", "糸魚川"],
            "キロ程": [58.51, 131.543],
            "緯度": [36.701322, 37.043119],
            "経度": [137.213608, 137.861307],
        }
    )

    result = btd.join_stop_station_location(train_df, location_df)

    assert result.row(0, named=True) == {
        "列車番号": "3500E",
        "停車駅名": "富山",
        "キロ程": pytest.approx(58.51),
        "緯度": pytest.approx(36.701322),
        "経度": pytest.approx(137.213608),
    }


def test_join_stop_station_location_preserves_row_count() -> None:
    train_df = pl.DataFrame(
        {
            "列車番号": ["3500E", "552E", "552E"],
            "停車駅名": ["富山", "富山", "糸魚川"],
        }
    )
    location_df = pl.DataFrame({"停車場名": ["富山", "糸魚川"], "キロ程": [58.51, 131.543]})

    result = btd.join_stop_station_location(train_df, location_df)

    assert result.height == train_df.height


def test_join_stop_station_location_unmatched_station_becomes_null() -> None:
    train_df = pl.DataFrame({"列車番号": ["3500E"], "停車駅名": ["未知駅"]})
    location_df = pl.DataFrame({"停車場名": ["富山"], "キロ程": [58.51]})

    result = btd.join_stop_station_location(train_df, location_df)

    assert result.height == 1
    assert result["キロ程"][0] is None
