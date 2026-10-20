"""scripts/eda_delhi_climate.py のテスト。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import eda_delhi_climate as eda
import pandas as pd
import pytest
from pandas_helpers import as_list, row_at


def _frame(dates: list[dt.date], pressure: list[float]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "date": [dt.datetime.combine(d, dt.time()) for d in dates],
            "meantemp": [20.0] * n,
            "humidity": [50.0] * n,
            "wind_speed": [5.0] * n,
            "meanpressure": pressure,
        }
    )


def test_date_continuity_detects_missing_and_duplicated_days() -> None:
    dates = [dt.date(2024, 1, 1), dt.date(2024, 1, 2), dt.date(2024, 1, 2), dt.date(2024, 1, 5)]
    summary = row_at(eda.date_continuity_summary(_frame(dates, [1000.0] * 4)), 0)
    assert summary["expected_days"] == 5  # 1/1〜1/5
    assert summary["missing_days"] == 2  # 1/3, 1/4
    assert summary["duplicated_days"] == 2  # 1/2 が2行


def test_out_of_range_values_lists_only_implausible_values() -> None:
    dates = [dt.date(2024, 1, d) for d in (1, 2, 3)]
    out = eda.out_of_range_values(_frame(dates, [1000.0, 7679.0, -3.0]), eda.PLAUSIBLE_RANGES)
    assert out["variable"].pipe(as_list) == ["meanpressure", "meanpressure"]
    assert out["value"].pipe(as_list) == [7679.0, -3.0]


def test_yearly_means_exclude_out_of_range_values() -> None:
    dates = [dt.date(2024, 1, d) for d in (1, 2, 3)]
    yearly = eda.yearly_means(_frame(dates, [1000.0, 1010.0, 7679.0]), eda.PLAUSIBLE_RANGES)
    row = row_at(yearly, 0)
    assert row["n_days"] == 3
    # 7679 hPa は除外して平均する
    assert row["meanpressure"] == pytest.approx(1005.0)


@pytest.mark.skipif(not eda.TRAIN_PATH.is_file(), reason="data/raw の生データが無い（git管理外）")
def test_eda_end_to_end(tmp_path: Path) -> None:
    eda.main(["--output-root", str(tmp_path)])
    tables = {p.name for p in (tmp_path / "tables").glob("*.csv")}
    assert {
        "delhi_climate_train__dataframe_overview.csv",
        "delhi_climate_train__column_overview.csv",
        "delhi_climate_train__date_continuity.csv",
        "delhi_climate_train__out_of_range_values.csv",
        "delhi_climate_train__yearly_means.csv",
    } <= tables
    figures = {p.name for p in (tmp_path / "figures").glob("*.png")}
    assert "delhi_climate_train__monthly_boxplots.png" in figures
    # 時系列診断: 4変数 × (推移・ACF・PACF)
    for variable in eda.VARIABLE_LABELS:
        for kind in ("series_with_moving_average", "acf_correlogram", "pacf_correlogram"):
            assert f"delhi_climate_train__{variable}__{kind}.png" in figures
    outliers = pd.read_csv(tmp_path / "tables" / "delhi_climate_train__out_of_range_values.csv")
    assert len(outliers) == 9  # 気圧の記録誤り9件
