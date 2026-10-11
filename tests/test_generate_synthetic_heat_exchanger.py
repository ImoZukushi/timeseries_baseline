"""scripts/generate_synthetic_heat_exchanger.py（熱交換器のダミーデータ）のテスト。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import generate_synthetic_heat_exchanger as gen
import numpy as np
import polars as pl
import pytest

HX = [f"HX{i:02d}" for i in range(1, 11)]
PRESSURES = ("入口圧力", "出口圧力")
TEMPERATURES = (
    "高温側熱交入口温度",
    "高温側熱交出口温度",
    "低温側熱交入口温度",
    "低温側熱交出口温度",
)


@pytest.fixture(scope="module")
def sim() -> gen.Simulation:
    return gen.simulate(seed=42)


@pytest.fixture(scope="module")
def frame(sim: gen.Simulation) -> pl.DataFrame:
    return sim.frame


def test_shape_columns_and_dates(frame: pl.DataFrame) -> None:
    expected = ["日付"] + [f"{h}_{c}" for h in HX for c in gen.COLUMNS]
    assert frame.columns == expected
    assert frame.width == 1 + 9 * 10
    # 温度の列は出力しない
    assert not any(col.endswith(TEMPERATURES) for col in frame.columns)
    dates = frame["日付"]
    assert (dates.min(), dates.max()) == (dt.date(2019, 1, 1), dt.date(2024, 12, 31))
    # 欠けのない日次（2019〜2024年、うるう年2回）
    assert frame.height == 365 * 6 + 2
    assert dates.diff().drop_nulls().dt.total_days().unique().to_list() == [1]


def test_dtypes_and_chemical_categories(frame: pl.DataFrame) -> None:
    for h in HX:
        for c in gen.COLUMNS:
            dtype = frame.schema[f"{h}_{c}"]
            assert dtype == (pl.Categorical if c == "薬剤種類" else pl.Float64), (h, c)
        chemicals = set(frame[f"{h}_薬剤種類"].cast(pl.String).to_list())
        assert chemicals <= set(gen.CHEMICALS)
    # 全体ではどの薬剤も使われている
    used = {v for h in HX for v in frame[f"{h}_薬剤種類"].cast(pl.String).to_list()}
    assert used == set(gen.CHEMICALS)


def test_crude_ratios_sum_to_one(frame: pl.DataFrame) -> None:
    for h in HX:
        a, b, c = (frame[f"{h}_{name}"] for name in gen.RATIO_COLUMNS)
        present = a.is_not_null()
        # 欠測は3列そろって欠ける
        assert (present == b.is_not_null()).all() and (present == c.is_not_null()).all()
        total = (a + b + c).filter(present)
        assert (total - 1.0).abs().max() < 1e-9
        for s in (a, b, c):
            assert s.drop_nulls().is_between(0.0, 1.0).all()


def test_stopped_days_have_zero_flow_and_missing_measurements(frame: pl.DataFrame) -> None:
    for h in HX:
        stopped = frame[f"{h}_低温側流量"] == 0
        assert stopped.sum() > 0
        assert (frame[f"{h}_高温側流量"].filter(stopped) == 0).all()
        for c in ("汚れ係数", *PRESSURES, *gen.RATIO_COLUMNS):
            assert frame[f"{h}_{c}"].filter(stopped).is_null().all(), (h, c)
    # 定期修理の期間は全基が停止している
    turnaround = frame.filter(pl.col("日付").is_between(*gen.TURNAROUND))
    assert turnaround.height == 21
    assert all((turnaround[f"{h}_低温側流量"] == 0).all() for h in HX)


def test_fouling_resets_after_cleaning(sim: gen.Simulation) -> None:
    frame, events = sim.frame, sim.events
    cleanings = events.filter(pl.col("種類") == "洗浄停止")
    assert cleanings.height > 0
    for row in cleanings.iter_rows(named=True):
        col = f"{row['機器']}_汚れ係数"
        before = frame.filter(pl.col("日付") < row["開始日"])[col].drop_nulls().tail(3).mean()
        after = frame.filter(pl.col("日付") > row["終了日"])[col].drop_nulls().head(3).mean()
        if before is None or after is None:
            continue  # データの端の洗浄
        assert after < before * 0.5, row


def test_physical_relationships(frame: pl.DataFrame) -> None:
    for h in HX:
        sub = pl.DataFrame(
            {
                "f": frame[f"{h}_汚れ係数"],
                "dp": frame[f"{h}_入口圧力"] - frame[f"{h}_出口圧力"],
            }
        ).drop_nulls()
        ranks = sub.select(pl.all().rank())
        # 汚れるほど圧力損失が増える（外れ値の影響を避けて順位相関）
        assert ranks.select(pl.corr("f", "dp")).item() > 0.5, h
        assert sub["dp"].median() > 0


def test_fouling_grows_faster_with_heavy_crude(frame: pl.DataFrame) -> None:
    # 原油C（汚れやすい）が多い日ほど、翌日の汚れ係数の増え方が大きい（全基をまとめて順位相関）
    parts = []
    for h in HX:
        parts.append(
            frame.select(
                pl.col(f"{h}_原油C比率").alias("c"),
                (pl.col(f"{h}_汚れ係数").shift(-7) - pl.col(f"{h}_汚れ係数")).alias("growth"),
            )
        )
    data = pl.concat(parts).drop_nulls()
    # 洗浄で下がった区間を除く
    data = data.filter(pl.col("growth") > -0.1)
    assert data.select(pl.corr(pl.col("c").rank(), pl.col("growth").rank())).item() > 0.1


def test_missing_values_and_outliers(frame: pl.DataFrame) -> None:
    for h in HX:
        running = frame[f"{h}_低温側流量"].fill_null(1.0) > 0
        for c in gen.SENSOR_COLUMNS:
            if "流量" in c:
                continue  # 流量の欠測は停止の 0 と区別するため、運転中の行で見る圧力だけを確認する
            rate = frame[f"{h}_{c}"].filter(running).null_count() / running.sum()
            assert 0.003 < rate < 0.06, (h, c, rate)
    # センサーの外れ値: 前後の日から大きく外れる値がある（目的変数には入れない）
    pressure = frame["HX05_入口圧力"].to_numpy()
    median = pl.Series(pressure).rolling_median(7, center=True, min_samples=3).to_numpy()
    ratio = np.abs(pressure / median - 1)
    assert np.nanmax(ratio) > 0.25
    fouling = frame["HX05_汚れ係数"]
    assert fouling.drop_nulls().min() >= 0.0


def test_simulation_is_reproducible() -> None:
    kwargs = {"start": dt.date(2019, 1, 1), "end": dt.date(2019, 3, 31)}
    a = gen.simulate(seed=7, **kwargs).frame
    assert a.equals(gen.simulate(seed=7, **kwargs).frame)
    assert not a.equals(gen.simulate(seed=8, **kwargs).frame)


def test_main_writes_csv_and_refuses_raw_dir(tmp_path: Path) -> None:
    gen.main(["--output-dir", str(tmp_path / "synthetic")])
    train = pl.read_csv(tmp_path / "synthetic" / "train.csv")
    assert train.shape == (2192, 91)
    assert (tmp_path / "synthetic" / "events.csv").is_file()
    with pytest.raises(ValueError, match="data/raw"):
        gen.main(["--output-dir", str(gen.data_dir() / "raw" / "synthetic")])


# --- 原油の性状（crude.csv） ---------------------------------------------------------------


def test_crude_properties_columns_and_types() -> None:
    crude = gen.crude_properties()
    assert crude.columns == [
        "原油",
        "密度",
        "比重",
        "API比重",
        "粘度",
        "全硫黄分",
        "金属分",
        "アスファルテン分",
        "炭化水素タイプ",
    ]
    # train.csv の原油比率の列（原油A比率など）と 原油 列で対応する
    assert crude["原油"].to_list() == [name[2] for name in gen.RATIO_COLUMNS]
    for col in ("密度", "比重", "API比重", "粘度", "全硫黄分", "アスファルテン分"):
        assert crude.schema[col] == pl.Float64, col
    assert crude.schema["金属分"] == pl.String
    assert isinstance(crude.schema["炭化水素タイプ"], pl.Enum)
    assert set(crude["炭化水素タイプ"].cast(pl.String)) <= set(gen.HYDROCARBON_TYPES)


def test_crude_gravity_is_consistent() -> None:
    crude = gen.crude_properties()
    for row in crude.iter_rows(named=True):
        # API比重 = 141.5 / 比重 − 131.5、密度 = 比重 × 15℃の水の密度
        assert 141.5 / row["比重"] - 131.5 == pytest.approx(row["API比重"], abs=0.05)
        assert row["密度"] == pytest.approx(row["比重"] * gen.WATER_DENSITY_15C, abs=1e-4)
        assert 0.0 < row["全硫黄分"] < 10.0 and 0.0 < row["アスファルテン分"] < 30.0


def test_crude_metals_are_combinations_of_one_to_three() -> None:
    for value in gen.crude_properties()["金属分"]:
        metals = value.split("・")
        assert 1 <= len(metals) <= 3
        assert len(set(metals)) == len(metals)
        assert set(metals) <= set(gen.METALS)


def test_crude_properties_match_fouling_order() -> None:
    # 汚れやすい原油ほど重質（API比重が小さく、粘度・硫黄分・アスファルテン分が多い）
    crude = gen.crude_properties()
    order = np.argsort(gen.CRUDE_FOULING)
    for col, increasing in [
        ("API比重", False),
        ("粘度", True),
        ("全硫黄分", True),
        ("アスファルテン分", True),
    ]:
        values = crude[col].to_numpy()[order]
        diffs = np.diff(values)
        assert (diffs > 0).all() if increasing else (diffs < 0).all(), col


def test_main_writes_crude_csv(tmp_path: Path) -> None:
    gen.main(["--output-dir", str(tmp_path)])
    crude = pl.read_csv(tmp_path / "crude.csv")
    assert crude.shape == (3, 9)
    assert crude["原油"].to_list() == ["A", "B", "C"]


# --- 薬剤の性状（chemical.csv） ------------------------------------------------------------


def test_chemical_properties_columns_and_values() -> None:
    chemical = gen.chemical_properties()
    assert chemical.columns == ["薬剤種類", "有効成分濃度", "ph", "比重", "溶解性"]
    for col in chemical.columns[1:]:
        assert chemical.schema[col] == pl.Float64, col
    for row in chemical.iter_rows(named=True):
        assert 0.0 < row["有効成分濃度"] <= 100.0
        assert 0.0 <= row["ph"] <= 14.0
        assert 0.5 < row["比重"] < 1.5
        assert row["溶解性"] > 0.0


def test_chemical_types_join_with_train(frame: pl.DataFrame) -> None:
    chemical = gen.chemical_properties()
    # train.csv の 薬剤種類 のうち「なし」以外と、値がそのまま一致する
    assert set(chemical["薬剤種類"]) == set(gen.CHEMICALS) - {"なし"}
    used = frame["HX01_薬剤種類"].cast(pl.String).rename("薬剤種類").to_frame()
    joined = used.join(chemical, on="薬剤種類", how="left")
    treated = joined.filter(pl.col("薬剤種類") != "なし")
    assert treated["有効成分濃度"].null_count() == 0


def test_chemical_properties_match_fouling_suppression() -> None:
    # 付着を強く抑える薬剤ほど、有効成分が濃く油によく溶ける
    chemical = gen.chemical_properties()
    factor = {name: gen.CHEMICAL_FACTOR[gen.CHEMICALS.index(name)] for name in chemical["薬剤種類"]}
    stronger = min(factor, key=factor.__getitem__)
    weaker = max(factor, key=factor.__getitem__)
    rows = {r["薬剤種類"]: r for r in chemical.iter_rows(named=True)}
    assert rows[stronger]["有効成分濃度"] > rows[weaker]["有効成分濃度"]
    assert rows[stronger]["溶解性"] > rows[weaker]["溶解性"]


def test_main_writes_chemical_csv(tmp_path: Path) -> None:
    gen.main(["--output-dir", str(tmp_path)])
    chemical = pl.read_csv(tmp_path / "chemical.csv")
    assert chemical.shape == (2, 5)
    assert chemical["薬剤種類"].to_list() == ["タイプA", "タイプB"]


# --- テスト期間（test.csv）------------------------------------------------------------------

SHORT = {"start": dt.date(2019, 1, 1), "train_end": dt.date(2019, 6, 30)}


@pytest.fixture(scope="module")
def train_test() -> gen.TrainTest:
    return gen.simulate_train_test(seed=42)


def test_train_part_is_identical_to_simulate() -> None:
    # テスト期間を続けて生成しても、学習期間（train.csv）の内容は変わらない
    both = gen.simulate_train_test(seed=3, test_end=dt.date(2019, 9, 30), **SHORT)
    alone = gen.simulate(seed=3, start=SHORT["start"], end=SHORT["train_end"])
    assert both.train.frame.equals(alone.frame)
    assert both.train.events.equals(alone.events)


def test_test_period_columns_and_target(train_test: gen.TrainTest) -> None:
    test = train_test.test
    assert test.columns == train_test.train.frame.columns
    assert (test["日付"].min(), test["日付"].max()) == (dt.date(2025, 1, 1), dt.date(2025, 12, 31))
    assert test.height == 365
    targets = [c for c in test.columns if c.endswith("_汚れ係数")]
    assert len(targets) == 10
    # 目的変数はすべて null、その他の列は値がある
    assert all(test[c].null_count() == test.height for c in targets)
    assert test["HX01_低温側流量"].null_count() < test.height
    # 正解は別の表（日付 ＋ 目的変数）
    answer = train_test.answer
    assert answer.columns == ["日付", *targets]
    assert answer["日付"].equals(test["日付"])
    assert answer["HX01_汚れ係数"].null_count() < answer.height


def test_test_period_continues_from_train(train_test: gen.TrainTest) -> None:
    train, answer, test = train_test.train.frame, train_test.answer, train_test.test
    for h in HX:
        last = train[f"{h}_汚れ係数"].drop_nulls().tail(3).mean()
        first = answer[f"{h}_汚れ係数"].drop_nulls().head(3).mean()
        assert last is not None and first is not None
        # 汚れ係数は途切れずにつながる（数日での増加と測定誤差の範囲）
        assert abs(first - last) < 0.15, (h, last, first)
        # 薬剤の期間も境目で引き継がれる
        assert train[f"{h}_薬剤種類"][-1] == test[f"{h}_薬剤種類"][0]
    # 原油比率の関係はテスト期間でも保たれる
    for h in HX:
        total = sum(test[f"{h}_{c}"] for c in gen.RATIO_COLUMNS).drop_nulls()
        assert (total - 1.0).abs().max() < 1e-9


def test_events_cover_both_periods(train_test: gen.TrainTest) -> None:
    events = train_test.events
    assert set(events["期間"]) == {"train", "test"}
    test_events = events.filter(pl.col("期間") == "test")
    assert test_events["開始日"].min() >= dt.date(2025, 1, 1)
    train_events = events.filter(pl.col("期間") == "train")
    assert train_events["終了日"].max() <= dt.date(2024, 12, 31)


def test_main_writes_test_files(tmp_path: Path) -> None:
    gen.main(["--output-dir", str(tmp_path)])
    test = pl.read_csv(tmp_path / "test.csv")
    answer = pl.read_csv(tmp_path / "test_answer.csv")
    assert test.shape == (365, 91)
    assert answer.shape == (365, 11)
    assert pl.read_csv(tmp_path / "events.csv").columns[-1] == "期間"
