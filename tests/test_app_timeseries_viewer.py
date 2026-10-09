"""app/timeseries_viewer（時系列ビューア）のテスト。"""

from __future__ import annotations

import datetime as dt
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from timeseries_viewer.catalog import discover_files, group_values, inspect_columns
from timeseries_viewer.figure import SeriesView, build_figure
from timeseries_viewer.series import (
    SeriesSpec,
    load_series,
    parse_axis_value,
    time_bounds,
    value_at,
    value_range,
)
from timeseries_viewer.storage import cache_paths, open_table

import util.csv_io as csv_io

MAIN = Path(__file__).resolve().parents[1] / "app" / "timeseries_viewer" / "main.py"


def _write_cp932(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("cp932"))
    return path


@pytest.fixture
def panel_csv(tmp_path: Path) -> Path:
    """2地点が混在するCP932の時間別データ（欠測記号 * を含む）。"""
    lines = ["年月日時,地点,気温(℃)"]
    for hour in range(48):
        t = dt.datetime(2016, 1, 1) + dt.timedelta(hours=hour)
        stamp = f"{t.year}/{t.month}/{t.day} {t.hour}:00"
        lines.append(f"{stamp},富山,{hour % 10}")
        lines.append(f"{stamp},金沢,{'*' if hour == 5 else -hour}")
    return _write_cp932(tmp_path / "data" / "raw" / "weather.csv", "\n".join(lines) + "\n")


# --- ファイル一覧・キャッシュ ----------------------------------------------------------------


def test_discover_files(tmp_path: Path) -> None:
    root = tmp_path / "data"
    for name in ("raw/a.csv", "raw/sub/b.parquet", "raw/c.txt", "cache/x.parquet"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x", encoding="utf-8")
    files = discover_files([root, tmp_path / "missing"], exclude=[root / "cache"])
    assert [f.label for f in files] == ["data/raw/a.csv", "data/raw/sub/b.parquet"]


def test_open_table_caches_csv_and_rebuilds_on_change(tmp_path: Path, panel_csv: Path) -> None:
    cache = tmp_path / "cache"
    original = panel_csv.read_bytes()
    lf = open_table(panel_csv, cache)
    parquet, meta = cache_paths(panel_csv, cache)
    assert parquet.is_file() and meta.is_file()
    frame = lf.collect()
    assert frame.schema["年月日時"] == pl.Datetime
    assert frame.height == 96
    # 欠測記号 * は欠損として読み込む
    assert frame["気温(℃)"].null_count() == 1
    # 2回目はキャッシュを使う（作り直さない）
    built = parquet.stat().st_mtime_ns
    open_table(panel_csv, cache).collect()
    assert parquet.stat().st_mtime_ns == built
    # 元ファイルは変更しない
    assert panel_csv.read_bytes() == original
    # 元ファイルが変わったら作り直す
    time.sleep(0.01)
    _write_cp932(panel_csv, "年月日時,地点,気温(℃)\n2016/1/1 0:00,富山,1\n")
    os.utime(panel_csv, ns=(time.time_ns(), time.time_ns()))
    assert open_table(panel_csv, cache).collect().height == 1


def test_open_table_reads_parquet_directly(tmp_path: Path) -> None:
    path = tmp_path / "x.parquet"
    pl.DataFrame({"t": [1, 2], "v": [0.5, 1.5]}).write_parquet(path)
    assert open_table(path, tmp_path / "cache").collect()["v"].to_list() == [0.5, 1.5]
    assert not (tmp_path / "cache").exists()


def test_scan_csv_auto_large_file_matches_read_csv_auto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = ["年月日時,風速(瞬時),風向(瞬時)"]
    for s in range(200):
        t = dt.datetime(2016, 1, 1) + dt.timedelta(seconds=s)
        lines.append(f"{t:%Y-%m-%d %H:%M:%S},{'*' if s == 3 else s / 10},{s % 360}")
    path = _write_cp932(tmp_path / "wind.csv", "\n".join(lines) + "\n")
    # 大容量ファイルの扱い（ヘッダ以外ASCIIなら lazy scan）を小さなファイルで再現する
    monkeypatch.setattr(csv_io, "LARGE_FILE_THRESHOLD_BYTES", 100)
    lazy = csv_io.scan_csv_auto(path, datetime_sample_rows=50).collect()
    eager = csv_io.read_csv_auto(path)
    assert lazy.equals(eager)
    assert lazy.schema["年月日時"] == pl.Datetime
    assert lazy["風速(瞬時)"].null_count() == 1


def test_inspect_columns_finds_group_column(tmp_path: Path, panel_csv: Path) -> None:
    lf = open_table(panel_csv, tmp_path / "cache")
    info = inspect_columns(lf)
    assert info.time_columns == ["年月日時"]
    assert info.value_columns == ["気温(℃)"]
    assert info.group_columns == ["地点"]
    assert info.default_group == "地点"
    assert group_values(lf, "地点") == ["富山", "金沢"]


def test_inspect_columns_numeric_time_axis() -> None:
    lf = pl.LazyFrame({"step": [1, 2, 3], "y": [0.1, 0.2, 0.3], "id": ["a", "a", "a"]})
    info = inspect_columns(lf)
    assert info.time_columns == ["step", "y"]
    # 時刻列（日時型）が無い場合はグループ列の既定を決めない
    assert info.default_group is None


# --- 系列 ------------------------------------------------------------------------------------


def _hourly(n: int = 1000) -> pl.LazyFrame:
    times = [dt.datetime(2020, 1, 1) + dt.timedelta(hours=i) for i in range(n)]
    values = np.sin(np.arange(n) / 10.0)
    values[n // 2] = 100.0  # 間引いても残るべきピーク
    return pl.LazyFrame({"t": times, "v": values, "g": ["a"] * n})


def test_load_series_filters_range_and_group(tmp_path: Path, panel_csv: Path) -> None:
    lf = open_table(panel_csv, tmp_path / "cache")
    spec = SeriesSpec("weather.csv", "年月日時", "気温(℃)", "地点", "金沢")
    data = load_series(lf, spec, (dt.datetime(2016, 1, 1, 10), dt.datetime(2016, 1, 1, 12)))
    assert data.columns == ["time", "value"]
    assert data["value"].to_list() == [-10.0, -11.0, -12.0]
    assert spec.label == "weather.csv / 気温(℃) / 地点=金沢"
    assert time_bounds(lf, spec) == (dt.datetime(2016, 1, 1), dt.datetime(2016, 1, 2, 23))
    assert value_range(lf, spec) == (-47.0, 0.0)


def test_load_series_downsamples_keeping_extremes() -> None:
    lf = _hourly()
    spec = SeriesSpec("x", "t", "v")
    data = load_series(lf, spec, max_points=100)
    assert data.height <= 100
    assert data["time"].is_sorted()
    assert data["value"].max() == 100.0
    assert data["value"].min() == pytest.approx(lf.collect()["v"].min())
    # 上限以下なら間引かない
    assert load_series(lf, spec, max_points=None).height == 1000


def test_load_series_date_column_becomes_datetime() -> None:
    lf = pl.LazyFrame({"d": [dt.date(2020, 1, 1), dt.date(2020, 1, 2)], "v": [1, 2]})
    data = load_series(lf, SeriesSpec("x", "d", "v"))
    assert data.schema["time"] == pl.Datetime("us")
    assert data.schema["value"] == pl.Float64


def test_value_at_returns_nearest_observation() -> None:
    lf = pl.LazyFrame(
        {
            "t": [dt.datetime(2020, 1, 1, h) for h in (0, 1, 2, 4)],
            "v": [0.0, 1.0, None, 4.0],
        }
    )
    spec = SeriesSpec("x", "t", "v")
    assert value_at(lf, spec, dt.datetime(2020, 1, 1, 0, 20)) == (dt.datetime(2020, 1, 1, 0), 0.0)
    assert value_at(lf, spec, dt.datetime(2020, 1, 1, 0, 40)) == (dt.datetime(2020, 1, 1, 1), 1.0)
    # 欠損の観測（2時）は飛ばして、値のある最も近い観測を返す
    assert value_at(lf, spec, dt.datetime(2020, 1, 1, 2)) == (dt.datetime(2020, 1, 1, 1), 1.0)
    # 範囲外でも端の観測を返す
    assert value_at(lf, spec, dt.datetime(2030, 1, 1)) == (dt.datetime(2020, 1, 1, 4), 4.0)
    empty = pl.LazyFrame(
        {"t": [dt.datetime(2020, 1, 1)], "v": [None]}, schema={"t": pl.Datetime, "v": pl.Float64}
    )
    assert value_at(empty, spec, dt.datetime(2020, 1, 1)) is None


def test_parse_axis_value() -> None:
    assert parse_axis_value("2016-01-01 12:30", True) == dt.datetime(2016, 1, 1, 12, 30)
    assert parse_axis_value("2016-01-01", True) == dt.datetime(2016, 1, 1)
    assert parse_axis_value("3.5", False) == 3.5
    with pytest.raises(ValueError):
        parse_axis_value("abc", True)


# --- 図 --------------------------------------------------------------------------------------


def test_build_figure_rows_cursor_and_axes() -> None:
    lf = _hourly(200)
    spec = SeriesSpec("x", "t", "v")
    data = load_series(lf, spec)
    cursor = dt.datetime(2020, 1, 3)
    views = [
        SeriesView("a", data, cursor_point=value_at(lf, spec, cursor)),
        SeriesView("b", data, y_range=(-2.0, 2.0)),
        SeriesView(
            "c", data.with_columns(pl.col("value").abs() + 1), y_range=(1.0, 100.0), log_y=True
        ),
    ]
    x_range = (dt.datetime(2020, 1, 2), dt.datetime(2020, 1, 5))
    fig = build_figure(views, x_range=x_range, cursor=cursor)
    layout = fig.layout
    # 3段で横軸を共有する（上の2段の横軸が一番下の段の横軸に連動する）
    assert [layout.xaxis.matches, layout.xaxis2.matches] == ["x3", "x3"]
    # カーソルの縦線は紙面全体（全段）にかかる
    (shape,) = layout.shapes
    assert (shape.yref, shape.y0, shape.y1, shape.x0) == ("paper", 0, 1, cursor)
    # カーソルの点（1段目のみ）と各段の折れ線
    assert len(fig.data) == 4
    assert tuple(layout.yaxis2.range) == (-2.0, 2.0)
    assert layout.yaxis3.type == "log"
    assert tuple(layout.yaxis3.range) == pytest.approx((0.0, 2.0))
    assert tuple(layout.xaxis.range) == x_range


# --- 画面 ------------------------------------------------------------------------------------


def test_app_runs_and_draws_selected_series(
    tmp_path: Path, panel_csv: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("TIMESERIES_VIEWER_ROOTS", str(tmp_path / "data"))
    monkeypatch.setenv("TIMESERIES_VIEWER_CACHE_DIR", str(tmp_path / "cache"))
    at = AppTest.from_file(str(MAIN), default_timeout=60).run()
    assert not at.exception
    at.selectbox(key="file_0").set_value("data/raw/weather.csv").run()
    assert not at.exception
    assert at.get("plotly_chart")
    # 既定のグループ列（地点）が選ばれ、値の選択肢が出る
    assert at.selectbox(key="group_0_data/raw/weather.csv").value == "地点"
    # カーソルを設定すると値の表が出る
    at.session_state["cursor"] = dt.datetime(2016, 1, 1, 3)
    at.run()
    assert not at.exception
    assert at.dataframe
    assert at.dataframe[0].value["値"].to_list() == [3.0]
