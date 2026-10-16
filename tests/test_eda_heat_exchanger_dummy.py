"""scripts/eda_heat_exchanger_dummy.py（熱交換器ダミーデータの EDA）のテスト。

データは legacy_heat_exchanger_data（旧形式のダミーデータ生成）で小さく作り、
一時ディレクトリに置いて使う。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import eda_heat_exchanger_dummy as eda
import legacy_heat_exchanger_data as gen
import polars as pl
import pytest


@pytest.fixture(scope="module")
def raw_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """1年分の学習期間と3か月分のテスト期間（data/raw/dummy と同じファイル構成）。"""
    root = tmp_path_factory.mktemp("dummy")
    result = gen.simulate_train_test(
        seed=1,
        start=dt.date(2019, 1, 1),
        train_end=dt.date(2019, 12, 31),
        test_end=dt.date(2020, 3, 31),
    )
    result.train.frame.write_csv(root / "train.csv")
    result.test.write_csv(root / "test.csv")
    result.answer.write_csv(root / "test_answer.csv")
    result.train.events.write_csv(root / "events.csv")
    gen.crude_properties().write_csv(root / "crude.csv")
    gen.chemical_properties().write_csv(root / "chemical.csv")
    return root


@pytest.fixture(scope="module")
def frames(raw_dir: Path) -> dict[str, pl.DataFrame]:
    return eda.load_dummy(raw_dir)


@pytest.fixture(scope="module")
def train_long(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return eda.to_long(frames["train"])


def test_to_long_shape_and_types(frames: dict[str, pl.DataFrame], train_long: pl.DataFrame) -> None:
    assert train_long.height == frames["train"].height * 10
    assert train_long.columns == ["日付", "機器", *eda.MEASUREMENTS, "圧力損失"]
    # 圧力損失 = 入口圧力 − 出口圧力
    row = train_long.drop_nulls(["入口圧力", "出口圧力"]).row(0, named=True)
    assert row["圧力損失"] == pytest.approx(row["入口圧力"] - row["出口圧力"])
    # テストの目的変数（全行が空）も Float64 になり、学習と同じ型で扱える
    test_long = eda.to_long(frames["test"])
    assert test_long.schema["汚れ係数"] == pl.Float64
    assert test_long["汚れ係数"].null_count() == test_long.height
    assert test_long.schema == train_long.schema


def test_file_overview_and_continuity(frames: dict[str, pl.DataFrame]) -> None:
    overview = eda.file_overview(frames)
    assert set(overview["file"]) == {f"{n}.csv" for n in frames}
    continuity = eda.date_continuity_summary(frames["train"], "train")
    assert continuity["missing_days"] == 0 and continuity["duplicated_days"] == 0
    assert continuity["expected_days"] == 365


def test_operation_summary_counts_stops(train_long: pl.DataFrame) -> None:
    summary = eda.operation_summary(train_long)
    assert summary["機器"].to_list() == eda.EXCHANGERS
    assert (summary["stop_days"] + summary["running_days"] == 365).all()
    # 停止日に測定値は残っていない
    assert (summary["stop_with_values"] == 0).all()
    # 流量の欠測は運転中として扱うので、低温側流量の欠測率は 0 より大きい
    assert summary["missing_低温側流量"].sum() > 0


def test_ratio_check(train_long: pl.DataFrame) -> None:
    check = eda.ratio_check(train_long).row(0, named=True)
    assert check["n_rows_partially_missing"] == 0
    assert check["max_abs_sum_error"] < 1e-9
    assert check["n_negative"] == 0 and check["n_above_one"] == 0
    # 1列だけ欠けた行は検出される
    broken = train_long.head(5).with_columns(pl.lit(None, dtype=pl.Float64).alias("原油A比率"))
    assert eda.ratio_check(broken)["n_rows_partially_missing"][0] > 0


def test_outlier_candidates_detects_spike() -> None:
    dates = [dt.datetime(2020, 1, 1) + dt.timedelta(days=i) for i in range(40)]
    values = [1.0] * 40
    values[20] = 2.0  # 前後の中央値の2倍
    long = pl.DataFrame(
        {
            "日付": dates,
            "機器": ["HX01"] * 40,
            "入口圧力": values,
            "低温側流量": [300.0] * 40,
        }
    )
    found = eda.outlier_candidates(long, columns=("入口圧力",))
    assert found["日付"].to_list() == [dates[20]]
    assert found["relative_gap"][0] == pytest.approx(1.0)


def test_fouling_growth_uses_horizon(
    frames: dict[str, pl.DataFrame], train_long: pl.DataFrame
) -> None:
    growth = eda.fouling_growth(train_long, frames["crude"])
    assert growth.height > 0
    assert growth["growth"].min() > eda.CLEANING_DROP / eda.GROWTH_HORIZON
    # 1日あたりの平均の増え方は正（汚れは時間とともにたまる）
    assert growth["growth"].mean() > 0
    by_chemical = eda.growth_by_chemical(growth, frames["chemical"])
    assert set(by_chemical["薬剤種類"]) <= {"なし", "タイプA", "タイプB"}
    # 薬剤の性状が結合される（なしは欠測）
    treated = by_chemical.filter(pl.col("薬剤種類") != "なし")
    assert treated["有効成分濃度"].null_count() == 0
    correlations = eda.growth_correlations(growth)
    assert correlations["機器"][0] == "全体"


def test_weighted_crude_properties(frames: dict[str, pl.DataFrame]) -> None:
    long = pl.DataFrame({"原油A比率": [1.0, 0.0], "原油B比率": [0.0, 0.5], "原油C比率": [0.0, 0.5]})
    crude = frames["crude"]
    out = eda.weighted_crude_properties(long, crude)
    asph = {r["原油"]: r["アスファルテン分"] for r in crude.iter_rows(named=True)}
    assert out["加重アスファルテン分"].to_list() == pytest.approx(
        [asph["A"], 0.5 * asph["B"] + 0.5 * asph["C"]]
    )


def test_events_summary(frames: dict[str, pl.DataFrame]) -> None:
    counts, intervals = eda.events_summary(frames["events"])
    assert counts["機器"].is_sorted()
    assert counts.drop("機器").sum_horizontal().sum() == frames["events"].height
    assert set(intervals.columns) >= {"n_cleanings", "mean_interval_days"}


def test_train_test_shift(frames: dict[str, pl.DataFrame], train_long: pl.DataFrame) -> None:
    shift = eda.train_test_shift(train_long, eda.to_long(frames["test"]))
    assert "圧力損失" in shift["column"].to_list()
    assert shift["standardized_mean_diff"].abs().max() < 3


def test_main_writes_tables_and_figures(raw_dir: Path, tmp_path: Path) -> None:
    eda.main(["--raw-dir", str(raw_dir), "--output-root", str(tmp_path)])
    tables = {p.name for p in (tmp_path / "tables").glob("dummy_hx__*.csv")}
    assert {
        "dummy_hx__file_overview.csv",
        "dummy_hx__operation_summary.csv",
        "dummy_hx__ratio_check.csv",
        "dummy_hx__outlier_candidates.csv",
        "dummy_hx__target_summary.csv",
        "dummy_hx__growth_by_chemical.csv",
        "dummy_hx__train_test_shift.csv",
    } <= tables
    figures = {p.name for p in (tmp_path / "figures").glob("dummy_hx*.png")}
    for name in (
        "fouling_timeseries",
        "fouling_vs_pressure_drop",
        "growth_drivers",
        "train_test_distributions",
    ):
        assert f"dummy_hx__{name}.png" in figures
    # src の汎用 EDA（品質確認・機器ごとの時系列診断）の図も出る
    assert any(name.startswith("dummy_hx_train_long__") for name in figures)
    assert any("dummy_hx_HX01__汚れ係数__acf_correlogram" in name for name in figures)
