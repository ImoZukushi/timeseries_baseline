"""scripts/generate_synthetic_heat_exchanger.py（A部門のダミーデータ生成）のテスト。

全期間（約7年）でも数秒で作れるが、テストは短い期間（学習2年・予測1年）で作って確認する。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import generate_synthetic_heat_exchanger as gen
import polars as pl
import pytest

START = dt.date(2019, 1, 1)
TRAIN_END = dt.date(2020, 12, 31)
TEST_END = dt.date(2021, 12, 31)
N_DAYS_ALL = (TEST_END - START).days + 2  # 予測期間の最終日の翌日まで


@pytest.fixture(scope="module")
def data() -> gen.SyntheticData:
    return gen.generate(seed=3, start=START, train_end=TRAIN_END, test_end=TEST_END)


@pytest.fixture(scope="module")
def frames(data: gen.SyntheticData) -> dict[str, pl.DataFrame]:
    return data.frames


def _between(frame: pl.DataFrame, column: str, low: float, high: float) -> bool:
    values = frame[column].drop_nulls()
    return bool(values.min() >= low and values.max() <= high)  # type: ignore[operator]


def test_files_columns_and_rows(frames: dict[str, pl.DataFrame]) -> None:
    assert set(frames) == set(gen.FILE_NAMES)
    n_ex = len(gen.EXCHANGERS)
    n_train = (TRAIN_END - START).days + 1
    n_test = (TEST_END - TRAIN_END).days
    assert frames["chemical"].columns == ["日付", "汚れ防止剤", "苛性ソーダ", "汚れ防止剤種類"]
    assert frames["chemical"].height == N_DAYS_ALL
    assert frames["target_train"].columns == list(gen.TARGET_COLUMNS)
    assert frames["target_train"].height == n_train * n_ex
    assert frames["target_test"].columns == ["ID", *gen.TARGET_COLUMNS]
    assert frames["target_test"].height == n_test * n_ex
    assert frames["target_test_answer"].columns == frames["target_test"].columns
    assert frames["exchanger_conditions"].columns == ["日付", "熱交換器", *gen.CONDITION_COLUMNS]
    assert frames["exchanger_conditions"].height == N_DAYS_ALL * n_ex
    assert frames["cleaning_stops"].columns == ["日付", "熱交換器", "洗浄日flg", "装置停止flg"]
    assert frames["cleaning_stops"].height == n_test * n_ex
    assert frames["crude_composition"].columns == ["日付", "原油コード", "原油配合比率"]
    assert frames["crude_properties"].columns == ["原油コード", *gen.CRUDE_PROPERTY_COLUMNS]
    assert len(gen.CRUDE_PROPERTY_COLUMNS) == 27
    assert frames["other_conditions"].columns == ["日付", *gen.OTHER_CONDITION_RANGES]
    assert frames["other_conditions"].height == N_DAYS_ALL
    # 熱交換器は指定の10基、日付は日次で途切れない
    assert frames["target_train"]["熱交換器"].unique().sort().to_list() == list(gen.EXCHANGERS)
    dates = frames["chemical"]["日付"]
    assert dates.min() == START and dates.max() == TEST_END + dt.timedelta(days=1)
    assert dates.diff().drop_nulls().dt.total_days().unique().to_list() == [1]


def test_value_ranges(frames: dict[str, pl.DataFrame]) -> None:
    chem = frames["chemical"]
    assert _between(chem, "汚れ防止剤", *gen.ANTIFOULANT_RANGE)
    assert _between(chem, "苛性ソーダ", *gen.CAUSTIC_RANGE)
    assert set(chem["汚れ防止剤種類"]) == {"A", "C"}
    # 注入を止めた日の計器の値は 0 付近（負になりうる）
    assert chem["汚れ防止剤"].min() < 0.05  # type: ignore[operator]
    cond = frames["exchanger_conditions"]
    for column, (low, high) in zip(gen.CONDITION_COLUMNS, gen.CONDITION_RANGES, strict=True):
        assert _between(cond, column, low, high)
        assert cond[column].null_count() > 0
    for column, (low, high) in gen.OTHER_CONDITION_RANGES.items():
        assert _between(frames["other_conditions"], column, low, high)
    train = frames["target_train"]
    assert _between(train, "汚れ係数", *gen.FOULING_RANGE)
    assert set(train["異常値flg"]) == {0, 1}


def test_crude_composition_sums_to_one(frames: dict[str, pl.DataFrame]) -> None:
    comp = frames["crude_composition"]
    assert set(comp["原油コード"]) <= set(gen.CRUDE_CODES)
    assert _between(comp, "原油配合比率", 0.0, 1.0)
    sums = comp.group_by("日付").agg(pl.col("原油配合比率").sum(), pl.len())
    assert sums.height == N_DAYS_ALL
    assert (sums["原油配合比率"] - 1.0).abs().max() < 1e-9  # type: ignore[operator]
    assert sums["len"].min() >= 3 and sums["len"].max() <= 5  # type: ignore[operator]


def test_crude_properties(frames: dict[str, pl.DataFrame]) -> None:
    props = frames["crude_properties"]
    assert props["原油コード"].to_list() == list(gen.CRUDE_CODES)
    specs = {**gen.CRUDE_PROPERTY_SPECS, **gen.VOLUME_SPECS, **gen.DENSITY_SPECS}
    for column, (low, high, _) in {**specs, **gen.FRACTION_SPECS}.items():
        assert _between(props, column, low, high), column
    total = props.select(pl.sum_horizontal(list(gen.VOLUME_SPECS))).to_series()
    assert (total - 100.0).abs().max() < 1e-6  # type: ignore[operator]
    # 重い原油ほど補正APIが低く、アスファルテンが多い（順位相関が負）
    rho = props.select(pl.corr("補正API", "アスファルテン[mass%]", method="spearman")).item()
    assert rho < -0.5


def test_prediction_period(frames: dict[str, pl.DataFrame]) -> None:
    test = frames["target_test"]
    assert test["ID"].to_list() == list(range(test.height))
    assert test["汚れ係数"].null_count() == test.height
    assert test["汚れ係数の翌日増加量"].null_count() == test.height
    assert test["日付"].min() == TRAIN_END + dt.timedelta(days=1)
    assert test["日付"].max() == TEST_END
    answer = frames["target_test_answer"]
    # 異常値flg と ID・日付・熱交換器は正解と同じ。正解には汚れ係数の値がある
    assert test.select("ID", "日付", "熱交換器", "異常値flg").equals(
        answer.select("ID", "日付", "熱交換器", "異常値flg")
    )
    assert answer["汚れ係数"].null_count() < answer.height


def test_increase_and_flags(frames: dict[str, pl.DataFrame]) -> None:
    targets = pl.concat([frames["target_train"], frames["target_test_answer"].drop("ID")])
    first = targets.filter(pl.col("日付") == START)
    # 初日の翌日増加量はある
    assert first["汚れ係数の翌日増加量"].null_count() == 0
    # 翌日増加量 = 翌日の観測値 − 当日の観測値（予測期間の初日をまたいでも同じ）
    checked = (
        targets.sort("熱交換器", "日付")
        .with_columns(pl.col("汚れ係数").shift(-1).over("熱交換器").alias("_next"))
        .drop_nulls(["汚れ係数", "_next", "汚れ係数の翌日増加量"])
    )
    diff = (checked["_next"] - checked["汚れ係数"] - checked["汚れ係数の翌日増加量"]).abs()
    assert diff.max() < 1e-9  # type: ignore[operator]
    # 欠測の行は必ず 1、欠測でない行にも 1（異常値）がある
    missing = pl.col("汚れ係数").is_null() | pl.col("汚れ係数の翌日増加量").is_null()
    assert (targets.filter(missing)["異常値flg"] == 1).all()
    assert targets.filter(~missing)["異常値flg"].sum() > 0


def test_stops_and_cleaning_records(data: gen.SyntheticData) -> None:
    frames, truth = data.frames, data.truth
    stops = frames["cleaning_stops"]
    # 洗浄の日は装置停止の日でもある
    assert stops.filter(pl.col("洗浄日flg") == 1)["装置停止flg"].min() == 1
    assert stops["装置停止flg"].sum() > 0
    # 停止日は運転条件・汚れ係数が欠測
    stopped = truth.filter(pl.col("停止の種類") > 0).select("日付", "熱交換器")
    assert stopped.height > 0
    cond = frames["exchanger_conditions"].join(stopped, on=["日付", "熱交換器"])
    assert (
        cond.select(pl.sum_horizontal(pl.all().exclude("日付", "熱交換器").is_not_null()))
        .sum()
        .item()
        == 0
    )
    target = frames["target_train"].join(stopped, on=["日付", "熱交換器"])
    assert target["汚れ係数"].null_count() == target.height
    # 学習期間にも洗浄はあるが、記録は予測期間だけ
    assert truth.filter((pl.col("日付") <= TRAIN_END) & (pl.col("停止の種類") == 1)).height > 0
    assert stops["日付"].min() > TRAIN_END


def test_physical_relationships() -> None:
    # 全期間で作り、付着速度と原油・薬剤の関係の向きを確かめる
    truth = gen.generate(seed=5).truth.filter(pl.col("付着速度") > 0)
    corr = truth.group_by("熱交換器").agg(
        pl.corr("付着速度", "原油の汚れやすさ", method="spearman").alias("crude"),
        pl.corr("付着速度", "汚れ防止剤_真値", method="spearman").alias("antifoulant"),
    )
    assert corr["crude"].median() > 0.2  # type: ignore[operator]
    assert corr["antifoulant"].median() < 0  # type: ignore[operator]
    # 汚れ係数の真値は洗浄で下がる（のこぎり形）
    drops = truth.sort("熱交換器", "日付").select(
        (pl.col("汚れ係数_真値").diff().over("熱交換器") < -0.002).sum()
    )
    assert drops.item() > 0


def test_reproducible() -> None:
    a = gen.generate(seed=7, start=START, train_end=dt.date(2019, 6, 30), test_end=TEST_END)
    b = gen.generate(seed=7, start=START, train_end=dt.date(2019, 6, 30), test_end=TEST_END)
    for key in gen.FILE_NAMES:
        assert a.frames[key].equals(b.frames[key]), key


def test_generate_rejects_invalid_period() -> None:
    with pytest.raises(ValueError, match="start < train_end < test_end"):
        gen.generate(start=TEST_END, train_end=TRAIN_END, test_end=START)


def test_main_writes_files(tmp_path: Path) -> None:
    gen.main(["--seed", "1", "--output-dir", str(tmp_path)])
    written = {p.name for p in tmp_path.glob("*.csv")}
    assert written == set(gen.FILE_NAMES.values())
    train = pl.read_csv(tmp_path / gen.FILE_NAMES["target_train"], try_parse_dates=True)
    assert train.schema["日付"] == pl.Date
    assert train["日付"].min() == gen.START and train["日付"].max() == gen.TRAIN_END


def test_main_refuses_raw_dir() -> None:
    with pytest.raises(ValueError, match="書き込めません"):
        gen.main(["--output-dir", str(gen.data_dir() / "raw" / "x")])
