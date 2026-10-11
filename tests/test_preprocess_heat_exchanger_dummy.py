"""scripts/preprocess_heat_exchanger_dummy.py（熱交換器ダミーデータの前処理）のテスト。

データは generate_synthetic_heat_exchanger で小さく作り、一時ディレクトリに置いて使う。
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import eda_heat_exchanger_dummy as eda
import generate_synthetic_heat_exchanger as gen
import polars as pl
import preprocess_heat_exchanger_dummy as prep
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
def combined(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    return prep.combine_periods(eda.to_long(frames["train"]), eda.to_long(frames["test"]))


@pytest.fixture(scope="module")
def cleaned(combined: pl.DataFrame) -> pl.DataFrame:
    return prep.clean(combined)


def _series(values: list[float | None], stopped: list[bool] | None = None) -> pl.DataFrame:
    """1基分の小さな縦持ちデータ（機器・日付の順）。"""
    n = len(values)
    return pl.DataFrame(
        {
            "日付": [dt.datetime(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)],
            "機器": ["HX01"] * n,
            "入口圧力": values,
            "停止": stopped or [False] * n,
        },
        schema_overrides={"入口圧力": pl.Float64},
    )


def test_combine_periods(frames: dict[str, pl.DataFrame], combined: pl.DataFrame) -> None:
    assert combined.height == (frames["train"].height + frames["test"].height) * 10
    assert set(combined["期間"]) == {"train", "test"}
    # 停止日は低温側流量が 0 の日
    assert combined.filter("停止")["低温側流量"].to_list() == [0.0] * combined["停止"].sum()
    with pytest.raises(ValueError, match="重なって"):
        prep.combine_periods(eda.to_long(frames["train"]), eda.to_long(frames["train"]))


def test_remove_outliers_sets_null_and_flag() -> None:
    values: list[float | None] = [1.0] * 40
    values[20] = 2.0
    out = prep.remove_outliers(_series(values), columns=("入口圧力",))
    assert out["入口圧力"].null_count() == 1 and out["入口圧力"][20] is None
    assert out["入口圧力_外れ値"].to_list() == [i == 20 for i in range(40)]


def test_remove_outliers_ignores_stopped_rows() -> None:
    # 停止日の 0 は中央値に入れず、外れ値ともみなさない
    values: list[float | None] = [1.0] * 40
    stopped = [False] * 40
    for i in range(15, 25):
        values[i], stopped[i] = 0.0, True
    out = prep.remove_outliers(_series(values, stopped), columns=("入口圧力",))
    assert not out["入口圧力_外れ値"].any()
    assert out["入口圧力"].null_count() == 0


def test_interpolate_short_gaps() -> None:
    values: list[float | None] = [float(i) for i in range(20)]
    for i in (3, 4):  # 2日の欠測 → 補間する
        values[i] = None
    for i in range(8, 13):  # 5日の欠測 → 長すぎるので残す
        values[i] = None
    values[19] = None  # 端の欠測 → 補間できない
    out = prep.interpolate_short_gaps(_series(values), ("入口圧力",), max_gap=3)
    assert out["入口圧力"][3] == pytest.approx(3.0) and out["入口圧力"][4] == pytest.approx(4.0)
    assert out["入口圧力"][8:13].null_count() == 5
    assert out["入口圧力"][19] is None
    assert out["入口圧力_補間"].sum() == 2


def test_interpolate_does_not_cross_stops() -> None:
    values: list[float | None] = [1.0, 1.0, None, None, None, 5.0, 5.0]
    stopped = [False, False, False, True, False, False, False]
    out = prep.interpolate_short_gaps(_series(values, stopped), ("入口圧力",), max_gap=3)
    # 停止日をはさむ欠測（区間の端）は補間しない。停止日自体も補間しない
    assert out["入口圧力"][2:5].null_count() == 3


def test_clean_summary(combined: pl.DataFrame, cleaned: pl.DataFrame) -> None:
    assert cleaned.height == combined.height
    assert {"原油比率_補間", *prep.OUTLIER_FLAGS} <= set(cleaned.columns)
    summary = prep.cleaning_summary(combined, cleaned)
    # クリーニングで運転中の欠測は減る（外れ値の除去で増えた分も補間で埋まる）
    for row in summary.iter_rows(named=True):
        if row["column"] != "圧力損失":
            assert row["missing_after"] <= row["missing_before"] + row["n_outliers"]
    # 補間後も原油比率の合計は 1
    ratios = cleaned.filter("原油比率_補間").select(pl.sum_horizontal(eda.RATIO_COLUMNS))
    assert ratios.to_series().to_list() == pytest.approx([1.0] * ratios.height)


def test_detect_cleanings_matches_events(
    frames: dict[str, pl.DataFrame], cleaned: pl.DataFrame
) -> None:
    cleanings = prep.detect_cleanings(cleaned)
    assert (cleanings["停止日数"] >= 1).all()
    check = {
        r["種類"]: r
        for r in prep.cleaning_detection_check(cleanings, frames["events"]).iter_rows(named=True)
    }
    # 洗浄の記録はすべて洗浄、突発停止はすべて洗浄なしと推定される
    for kind in ("洗浄停止", "定期修理（洗浄）"):
        if kind in check:
            assert check[kind]["n_detected_as_cleaning"] == check[kind]["n_periods"]
    if "突発停止" in check:
        assert check["突発停止"]["n_detected_as_cleaning"] == 0


def test_add_cycle_features() -> None:
    n = 12
    stopped = [False] * n
    stopped[4] = stopped[5] = True
    long = pl.DataFrame(
        {
            "日付": [dt.datetime(2020, 1, 1) + dt.timedelta(days=i) for i in range(n)],
            "機器": ["HX01"] * n,
            "停止": stopped,
            "圧力損失": [float(i) for i in range(n)],
            "加重アスファルテン分": [1.0] * n,
            "薬剤種類": ["なし", "タイプA"] * (n // 2),
        }
    )
    cleanings = pl.DataFrame(
        {
            "機器": ["HX01"],
            "開始日": [dt.datetime(2020, 1, 5)],
            "終了日": [dt.datetime(2020, 1, 6)],
            "洗浄推定": [True],
        }
    )
    out = prep.add_cycle_features(long, cleanings, baseline_days=2)
    assert out["洗浄回数"].to_list() == [0] * 6 + [1] * 6
    # 最初の洗浄より前は null
    assert out["洗浄後日数"][:6].null_count() == 6
    cycle = out.filter(pl.col("洗浄回数") == 1)
    assert cycle["洗浄後日数"].to_list() == [1, 2, 3, 4, 5, 6]
    assert cycle["洗浄後累積加重アスファルテン分"].to_list() == pytest.approx([1, 2, 3, 4, 5, 6])
    # 薬剤なしは 6, 8, 10 番目の日（サイクルの 1, 3, 5 日目）
    assert cycle["洗浄後薬剤なし日数"].to_list() == [1, 1, 2, 2, 3, 3]
    # 洗浄直後圧力損失: 最初の2運転日（6, 7）の当日までの平均
    assert cycle["洗浄直後圧力損失"].to_list() == pytest.approx([6.0, 6.5, 6.5, 6.5, 6.5, 6.5])


def test_build_and_split(frames: dict[str, pl.DataFrame], cleaned: pl.DataFrame) -> None:
    cleanings = prep.detect_cleanings(cleaned)
    features = prep.build_features(cleaned, cleanings, frames["crude"], frames["chemical"])
    assert features.height == cleaned.height
    train, test = prep.split_datasets(features)
    assert train[eda.TARGET].null_count() == 0
    assert not train["停止"].any()
    assert test.height == frames["test"].height * 10
    assert test[eda.TARGET].null_count() == test.height
    assert train.columns == test.columns and "期間" not in train.columns
    # 薬剤なしの有効成分濃度は 0、薬剤ありは chemical.csv の値
    assert (train.filter(pl.col("薬剤種類") == "なし")["薬剤有効成分濃度"] == 0).all()
    assert train["機器番号"].min() == 1 and train["機器番号"].max() == 10
    # テスト期間の初日も、学習期間からサイクルを引き継ぐ（洗浄済みの機器は洗浄後日数がある）
    first_test = test.filter(pl.col("日付") == test["日付"].min())
    assert first_test.filter(pl.col("洗浄回数") > 0)["洗浄後日数"].null_count() == 0
    columns = prep.dataset_columns(train, test)
    assert columns["column"].to_list() == train.columns
    correlations = prep.feature_correlations(train)
    assert eda.TARGET not in correlations["feature"].to_list()


def test_main_writes_outputs(raw_dir: Path, tmp_path: Path) -> None:
    prep.main(
        [
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(tmp_path / "processed"),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    train = pl.read_parquet(tmp_path / "processed" / "train_dataset.parquet")
    test = pl.read_parquet(tmp_path / "processed" / "test_dataset.parquet")
    assert train.schema == test.schema
    tables = {p.name for p in (tmp_path / "out" / "tables").glob("*.csv")}
    assert {
        "dummy_hx_prep__cleaning_summary.csv",
        "dummy_hx_prep__stop_periods.csv",
        "dummy_hx_prep__cleaning_detection_check.csv",
        "dummy_hx_prep__dataset_columns.csv",
        "dummy_hx_prep__feature_correlations.csv",
    } <= tables
    assert list((tmp_path / "out" / "figures").glob("dummy_hx_prep__cleaning_detection_HX*.png"))


def test_main_refuses_raw_output_dir(raw_dir: Path) -> None:
    with pytest.raises(ValueError, match="書き込めません"):
        prep.main(["--raw-dir", str(raw_dir), "--output-dir", str(prep.RAW_DIR / "x")])
