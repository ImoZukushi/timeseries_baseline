"""csv_quality モジュールのテスト。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pytest

from analysis_project import csv_quality as cq


def _write_cp932(path: Path, content: str) -> None:
    path.write_bytes(content.encode("cp932"))


def _write_utf8(path: Path, content: str) -> None:
    path.write_bytes(content.encode("utf-8"))


# --- detect_encoding -------------------------------------------------------


def test_detect_encoding_utf8(tmp_path: Path) -> None:
    path = tmp_path / "utf8.csv"
    _write_utf8(path, "a,b\n1,あいう\n")
    assert cq.detect_encoding(path) == "utf-8"


def test_detect_encoding_cp932(tmp_path: Path) -> None:
    path = tmp_path / "cp932.csv"
    _write_cp932(path, "年月日時,風速\n2020-01-01,1.1\n")
    assert cq.detect_encoding(path) == "cp932"


# --- read_csv_auto -----------------------------------------------------------


def test_read_csv_auto_small_cp932(tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    _write_cp932(path, "地点,気温\n富山,9.4\n高田,8.2\n")
    df = cq.read_csv_auto(path)
    assert df.columns == ["地点", "気温"]
    assert df["地点"].to_list() == ["富山", "高田"]


def test_read_csv_auto_large_ascii_data_uses_lossy_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 大容量ファイル判定の閾値を下げて、lazy scan（utf8-lossy）経路を強制する。
    monkeypatch.setattr(cq, "LARGE_FILE_THRESHOLD_BYTES", 10)
    path = tmp_path / "large_ascii.csv"
    header = "年月日時,風速(瞬時),風向(瞬時)\n"
    rows = "\n".join(f"2020-01-01 00:00:{i:02d},{i * 0.1:.1f},{i}" for i in range(100))
    _write_cp932(path, header + rows + "\n")

    df = cq.read_csv_auto(path)

    assert df.columns == ["年月日時", "風速(瞬時)", "風向(瞬時)"]
    assert df.height == 100
    assert df["風向(瞬時)"].to_list()[:3] == [0, 1, 2]


def test_read_csv_auto_large_non_ascii_data_falls_back_to_full_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cq, "LARGE_FILE_THRESHOLD_BYTES", 10)
    path = tmp_path / "large_non_ascii.csv"
    header = "地点,気温\n"
    rows = "\n".join(["富山,9.4", "高田,8.2"] * 5)
    _write_cp932(path, header + rows + "\n")

    df = cq.read_csv_auto(path)

    assert df.columns == ["地点", "気温"]
    assert set(df["地点"].to_list()) == {"富山", "高田"}


def test_read_csv_auto_parses_datetime_like_columns(tmp_path: Path) -> None:
    path = tmp_path / "datetime.csv"
    _write_utf8(path, "ts,value\n2020-01-01 00:00:00,1\n2020-01-02 00:00:00,2\n")
    df = cq.read_csv_auto(path)
    assert df.schema["ts"] == pl.Datetime


# --- dataframe_overview / column_overview -----------------------------------


def test_dataframe_overview_counts_duplicates_and_missing() -> None:
    df = pl.DataFrame({"a": [1, 1, 2, None], "b": ["x", "x", "y", "z"]})
    overview = cq.dataframe_overview(df, "sample", "sample.csv")
    row = overview.row(0, named=True)

    assert row["n_rows"] == 4
    assert row["n_columns"] == 2
    assert row["n_duplicated_rows"] == 1  # ("a"=1,"b"="x") が2件重複
    assert row["n_columns_with_missing"] == 1
    assert row["total_missing_cells"] == 1


def test_dataframe_overview_empty_dataframe() -> None:
    df = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})
    overview = cq.dataframe_overview(df, "empty", "empty.csv")
    row = overview.row(0, named=True)
    assert row["n_rows"] == 0
    assert row["duplicated_row_rate"] is None


def test_column_overview_numeric_stats() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, None]})
    overview = cq.column_overview(df, "sample")
    row = overview.row(0, named=True)

    assert row["is_numeric"] is True
    assert row["n_missing"] == 1
    assert row["missing_rate"] == pytest.approx(0.2)
    assert row["mean"] == pytest.approx(2.5)
    assert row["min"] == pytest.approx(1.0)
    assert row["max"] == pytest.approx(4.0)


def test_column_overview_categorical_mode() -> None:
    df = pl.DataFrame({"c": ["a", "b", "b", "b", "c"]})
    overview = cq.column_overview(df, "sample")
    row = overview.row(0, named=True)

    assert row["is_numeric"] is False
    assert row["mode"] == "b"
    assert row["mode_rate"] == pytest.approx(3 / 5)
    assert row["n_unique"] == 3


# --- plotting -----------------------------------------------------------


def test_plot_missing_overview_skips_when_no_missing(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    created = cq.plot_missing_overview(df, "sample", 3, tmp_path / "out.png")
    assert created is False
    assert not (tmp_path / "out.png").exists()


def test_plot_missing_overview_creates_file_when_missing(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1, None, 3]})
    out_path = tmp_path / "missing.png"
    created = cq.plot_missing_overview(df, "sample", 3, out_path)
    assert created is True
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_numeric_histograms_skips_without_numeric_columns(tmp_path: Path) -> None:
    df = pl.DataFrame({"c": ["a", "b", "c"]})
    created = cq.plot_numeric_histograms(df, "sample", 3, tmp_path / "out.png")
    assert created is False


def test_plot_numeric_histograms_creates_file(tmp_path: Path) -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    out_path = tmp_path / "hist.png"
    created = cq.plot_numeric_histograms(df, "sample", 4, out_path)
    assert created is True
    assert out_path.exists()


def test_plot_categorical_top_values_excludes_high_cardinality(tmp_path: Path) -> None:
    df = pl.DataFrame({"id": [str(i) for i in range(20)], "cat": ["x"] * 10 + ["y"] * 10})
    out_path = tmp_path / "cat.png"
    created = cq.plot_categorical_top_values(df, "sample", 20, out_path, max_unique_rate=0.5)
    assert created is True
    assert out_path.exists()


def test_plot_categorical_top_values_skips_when_all_high_cardinality(tmp_path: Path) -> None:
    df = pl.DataFrame({"id": [str(i) for i in range(20)]})
    created = cq.plot_categorical_top_values(df, "sample", 20, tmp_path / "out.png")
    assert created is False


def test_plot_correlation_heatmap_requires_two_numeric_columns(tmp_path: Path) -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    created = cq.plot_correlation_heatmap(df, "sample", 3, tmp_path / "out.png")
    assert created is False


def test_plot_correlation_heatmap_creates_file(tmp_path: Path) -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    out_path = tmp_path / "corr.png"
    created = cq.plot_correlation_heatmap(df, "sample", 3, out_path)
    assert created is True
    assert out_path.exists()


def test_plot_correlation_heatmap_creates_file_with_nulls(tmp_path: Path) -> None:
    # 欠損値を含んでいても（polarsのcorr()がNaNだけを返す状態にならず）図を作成できる
    df = pl.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0], "y": [2.0, 4.0, 6.0, None, 10.0]})
    out_path = tmp_path / "corr_with_nulls.png"
    created = cq.plot_correlation_heatmap(df, "sample", 5, out_path)
    assert created is True
    assert out_path.exists()


def test_plot_correlation_heatmap_false_when_too_few_complete_rows(tmp_path: Path) -> None:
    df = pl.DataFrame({"x": [1.0, None, None], "y": [None, 2.0, None]})
    created = cq.plot_correlation_heatmap(df, "sample", 3, tmp_path / "out.png")
    assert created is False


# --- complete_case_correlation ---------------------------------------------


def test_complete_case_correlation_drops_rows_with_any_null() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, None, 4.0], "y": [2.0, 4.0, 6.0, None]})
    corr, n_excluded = cq.complete_case_correlation(df, ["x", "y"])
    assert n_excluded == 2
    assert corr["x"][1] == pytest.approx(1.0)  # x,yが完全に線形（残り2行）なので相関1.0


def test_complete_case_correlation_no_nulls_excludes_nothing() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]})
    corr, n_excluded = cq.complete_case_correlation(df, ["x", "y"])
    assert n_excluded == 0
    assert corr["x"][1] == pytest.approx(-1.0)


def test_plot_record_pattern_creates_file(tmp_path: Path) -> None:
    df = pl.DataFrame({"x": list(range(100))})
    out_path = tmp_path / "pattern.png"
    created = cq.plot_record_pattern(df, "sample", 100, out_path, max_points=10)
    assert created is True
    assert out_path.exists()


# --- naming / end-to-end -----------------------------------------------------


def test_make_dataset_name_joins_subdirectory() -> None:
    raw_dir = Path("data/raw")
    path = raw_dir / "wind_0" / "下条川.csv"
    assert cq.make_dataset_name(path, raw_dir) == "wind_0_下条川"


def test_run_quality_checks_end_to_end(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    figures_dir = tmp_path / "figures"
    tables_dir = tmp_path / "tables"

    path = raw_dir / "sample.csv"
    _write_utf8(path, "a,b\n1,x\n2,y\n2,y\n,z\n")

    dataset_name = cq.make_dataset_name(path, raw_dir)
    df = cq.read_csv_auto(path)
    df_overview, col_overview = cq.run_quality_checks(
        df, dataset_name, str(path.relative_to(raw_dir)), figures_dir, tables_dir
    )

    assert df_overview.row(0, named=True)["n_rows"] == 4
    assert col_overview.height == 2
    assert (tables_dir / "sample__dataframe_overview.csv").exists()
    assert (tables_dir / "sample__column_overview.csv").exists()
    assert (figures_dir / "sample__missing_overview.png").exists()
    assert (figures_dir / "sample__numeric_histograms.png").exists()


# --- chunk_numeric_columns ---------------------------------------------------


def test_chunk_numeric_columns_empty_list() -> None:
    assert cq.chunk_numeric_columns([]) == []


def test_chunk_numeric_columns_fewer_than_max() -> None:
    assert cq.chunk_numeric_columns(["a", "b"], max_per_chunk=3) == [["a", "b"]]


def test_chunk_numeric_columns_exact_multiple() -> None:
    columns = ["a", "b", "c", "d", "e", "f"]
    assert cq.chunk_numeric_columns(columns, max_per_chunk=3) == [
        ["a", "b", "c"],
        ["d", "e", "f"],
    ]


def test_chunk_numeric_columns_with_remainder() -> None:
    columns = ["a", "b", "c", "d"]
    assert cq.chunk_numeric_columns(columns, max_per_chunk=3) == [["a", "b", "c"], ["d"]]


# --- plot_scatter_matrix ------------------------------------------------------


def test_plot_scatter_matrix_requires_nonempty_row_and_col(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0]})
    created = cq.plot_scatter_matrix(df, [], ["a"], "sample", 3, tmp_path / "out.png")
    assert created is False
    assert not (tmp_path / "out.png").exists()

    created = cq.plot_scatter_matrix(df, ["a"], [], "sample", 3, tmp_path / "out2.png")
    assert created is False


def test_plot_scatter_matrix_creates_file_for_two_columns(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
    out_path = tmp_path / "scatter2.png"
    created = cq.plot_scatter_matrix(df, ["a", "b"], ["a", "b"], "sample", 4, out_path)
    assert created is True
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_plot_scatter_matrix_creates_file_for_three_columns(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 1.0, 2.0, 2.0],
        }
    )
    out_path = tmp_path / "scatter3.png"
    created = cq.plot_scatter_matrix(
        df, ["a", "b", "c"], ["a", "b", "c"], "sample", 4, out_path
    )
    assert created is True
    assert out_path.exists()


def test_plot_scatter_matrix_creates_file_for_single_cell_cross_block(tmp_path: Path) -> None:
    # 行1列×列1列（異なるカラム同士）でも1セルの散布図として作成できる
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    out_path = tmp_path / "scatter1x1.png"
    created = cq.plot_scatter_matrix(df, ["a"], ["b"], "sample", 3, out_path)
    assert created is True
    assert out_path.exists()


def test_plot_scatter_matrix_creates_file_for_rectangular_block(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 1.0, 2.0, 2.0],
            "d": [2.0, 2.0, 3.0, 3.0],
        }
    )
    out_path = tmp_path / "scatter_rect.png"
    created = cq.plot_scatter_matrix(df, ["a", "b"], ["c", "d"], "sample", 4, out_path)
    assert created is True
    assert out_path.exists()


def test_plot_scatter_matrix_handles_nulls_without_error(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0], "b": [4.0, 3.0, None, 1.0]})
    out_path = tmp_path / "scatter_nulls.png"
    created = cq.plot_scatter_matrix(df, ["a", "b"], ["a", "b"], "sample", 4, out_path)
    assert created is True
    assert out_path.exists()


def test_plot_scatter_matrix_diagonal_only_when_row_col_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # row=[a,b], col=[a,c] -> 縦横で一致するのは "a" のセルだけなので、
    # ヒストグラムは1回だけ、残り3セルは散布図になるはず。
    hist_call_count = 0
    scatter_call_count = 0
    original_hist = plt.Axes.hist
    original_scatter = plt.Axes.scatter

    def fake_hist(self: plt.Axes, *args: object, **kwargs: object) -> object:
        nonlocal hist_call_count
        hist_call_count += 1
        return original_hist(self, *args, **kwargs)

    def fake_scatter(self: plt.Axes, *args: object, **kwargs: object) -> object:
        nonlocal scatter_call_count
        scatter_call_count += 1
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "hist", fake_hist)
    monkeypatch.setattr(plt.Axes, "scatter", fake_scatter)

    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0], "c": [1.0, 1.0, 2.0]})
    cq.plot_scatter_matrix(df, ["a", "b"], ["a", "c"], "sample", 3, tmp_path / "out.png")

    assert hist_call_count == 1
    assert scatter_call_count == 3


def test_plot_scatter_matrix_no_diagonal_when_rows_and_cols_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hist_call_count = 0
    original_hist = plt.Axes.hist

    def fake_hist(self: plt.Axes, *args: object, **kwargs: object) -> object:
        nonlocal hist_call_count
        hist_call_count += 1
        return original_hist(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "hist", fake_hist)

    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [3.0, 2.0, 1.0],
            "c": [1.0, 1.0, 2.0],
            "d": [2.0, 2.0, 1.0],
        }
    )
    cq.plot_scatter_matrix(df, ["a", "b"], ["c", "d"], "sample", 3, tmp_path / "out.png")

    assert hist_call_count == 0


def test_plot_scatter_matrix_uses_default_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_alphas: list[float | None] = []
    original_scatter = plt.Axes.scatter

    def fake_scatter(self: plt.Axes, *args: object, **kwargs: object) -> object:
        captured_alphas.append(kwargs.get("alpha"))  # type: ignore[arg-type]
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "scatter", fake_scatter)
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    cq.plot_scatter_matrix(df, ["a", "b"], ["a", "b"], "sample", 3, tmp_path / "out.png")

    assert captured_alphas
    assert all(a == pytest.approx(0.3) for a in captured_alphas)


def test_plot_scatter_matrix_respects_custom_alpha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_alphas: list[float | None] = []
    original_scatter = plt.Axes.scatter

    def fake_scatter(self: plt.Axes, *args: object, **kwargs: object) -> object:
        captured_alphas.append(kwargs.get("alpha"))  # type: ignore[arg-type]
        return original_scatter(self, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "scatter", fake_scatter)
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
    cq.plot_scatter_matrix(
        df, ["a", "b"], ["a", "b"], "sample", 3, tmp_path / "out.png", alpha=0.7
    )

    assert captured_alphas
    assert all(a == pytest.approx(0.7) for a in captured_alphas)


# --- plot_scatter_matrices -----------------------------------------------------


def test_plot_scatter_matrices_requires_two_numeric_columns(tmp_path: Path) -> None:
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "c": ["x", "y", "z"]})
    created = cq.plot_scatter_matrices(df, "sample", 3, tmp_path)
    assert created == []


def test_plot_scatter_matrices_covers_all_combinations_without_omission(
    tmp_path: Path,
) -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 1.0, 2.0, 2.0],
            "d": [2.0, 2.0, 3.0, 3.0],
            "e": [5.0, 4.0, 3.0, 2.0],
        }
    )
    created = cq.plot_scatter_matrices(df, "sample", 4, tmp_path, max_cols_per_image=3)
    # 5列 -> [a,b,c](3列) と [d,e](2列)。総当たりの漏れがないよう
    # (3x3の対角), (3x2の交差), (2x2の対角) の3枚が作成される。
    assert len(created) == 3
    names = {p.name for p in created}
    assert "sample__scatter_matrix__a_b_c.png" in names
    assert "sample__scatter_matrix__a_b_c__x__d_e.png" in names
    assert "sample__scatter_matrix__d_e.png" in names
    for path in created:
        assert path.exists()


def test_plot_scatter_matrices_lone_trailing_column_still_covered_via_cross_block(
    tmp_path: Path,
) -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0],
            "c": [1.0, 1.0, 2.0, 2.0],
            "d": [2.0, 2.0, 3.0, 3.0],
        }
    )
    created = cq.plot_scatter_matrices(df, "sample", 4, tmp_path, max_cols_per_image=3)
    # 4列 -> [a,b,c](3列) と [d](1列)。
    # dだけの自己組み合わせ(1x1)は対象外だが、[a,b,c]×[d]の交差(3x1)でdの関係は網羅される。
    names = {p.name for p in created}
    assert names == {
        "sample__scatter_matrix__a_b_c.png",
        "sample__scatter_matrix__a_b_c__x__d.png",
    }


def test_plot_scatter_matrices_sanitizes_unsafe_column_names_in_filename(tmp_path: Path) -> None:
    df = pl.DataFrame({"a/b": [1.0, 2.0, 3.0], "c:d": [3.0, 2.0, 1.0]})
    created = cq.plot_scatter_matrices(df, "sample", 3, tmp_path)
    assert len(created) == 1
    assert created[0].exists()
    assert "/" not in created[0].name
    assert ":" not in created[0].name
