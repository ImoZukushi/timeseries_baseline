"""util パッケージ（paths / plotting / csv_io）のテスト。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import pytest

from util import csv_io, paths, plotting


def _write_cp932(path: Path, content: str) -> None:
    path.write_bytes(content.encode("cp932"))


def _write_utf8(path: Path, content: str) -> None:
    path.write_bytes(content.encode("utf-8"))


# --- paths ---------------------------------------------------------------


def test_get_repo_root_contains_pyproject_toml() -> None:
    root = paths.get_repo_root()
    assert (root / "pyproject.toml").exists()


def test_data_dir_is_under_repo_root() -> None:
    assert paths.data_dir() == paths.get_repo_root() / "data"


def test_outputs_dir_is_under_repo_root() -> None:
    assert paths.outputs_dir() == paths.get_repo_root() / "outputs"


def test_ensure_parent_dir_creates_missing_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c.txt"
    assert not target.parent.exists()

    result = paths.ensure_parent_dir(target)

    assert result == target
    assert target.parent.exists()


def test_sanitize_filename_component_replaces_unsafe_chars() -> None:
    assert paths.sanitize_filename_component('a:b/c\\d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"


def test_sanitize_filename_component_leaves_safe_text_unchanged() -> None:
    assert (
        paths.sanitize_filename_component("wind_0_下条川__風速(瞬時)")
        == "wind_0_下条川__風速(瞬時)"
    )


# --- plotting --------------------------------------------------------------


def test_ensure_japanese_font_sets_meiryo() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plotting.ensure_japanese_font()
    assert plt.rcParams["font.family"] == ["Meiryo"]
    assert plt.rcParams["axes.unicode_minus"] is False


def test_add_caption_adds_text_to_figure() -> None:
    fig, _ax = plt.subplots()
    try:
        before = len(fig.texts)
        plotting.add_caption(fig, "対象: sample — テストキャプション。")
        assert len(fig.texts) == before + 1
        assert fig.texts[-1].get_text() == "対象: sample — テストキャプション。"
    finally:
        plt.close(fig)


# --- csv_io ------------------------------------------------------------------


def test_detect_encoding_utf8(tmp_path: Path) -> None:
    path = tmp_path / "utf8.csv"
    _write_utf8(path, "a,b\n1,あいう\n")
    assert csv_io.detect_encoding(path) == "utf-8"


def test_detect_encoding_cp932(tmp_path: Path) -> None:
    path = tmp_path / "cp932.csv"
    _write_cp932(path, "年月日時,風速\n2020-01-01,1.1\n")
    assert csv_io.detect_encoding(path) == "cp932"


def test_read_csv_auto_small_cp932(tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    _write_cp932(path, "地点,気温\n富山,9.4\n高田,8.2\n")
    df = csv_io.read_csv_auto(path)
    assert df.columns == ["地点", "気温"]
    assert df["地点"].to_list() == ["富山", "高田"]


def test_read_csv_auto_large_ascii_data_uses_lossy_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 大容量ファイル判定の閾値を下げて、lazy scan（utf8-lossy）経路を強制する。
    monkeypatch.setattr(csv_io, "LARGE_FILE_THRESHOLD_BYTES", 10)
    path = tmp_path / "large_ascii.csv"
    header = "年月日時,風速(瞬時),風向(瞬時)\n"
    rows = "\n".join(f"2020-01-01 00:00:{i:02d},{i * 0.1:.1f},{i}" for i in range(100))
    _write_cp932(path, header + rows + "\n")

    df = csv_io.read_csv_auto(path)

    assert df.columns == ["年月日時", "風速(瞬時)", "風向(瞬時)"]
    assert df.height == 100
    assert df["風向(瞬時)"].to_list()[:3] == [0, 1, 2]


def test_read_csv_auto_large_non_ascii_data_falls_back_to_full_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(csv_io, "LARGE_FILE_THRESHOLD_BYTES", 10)
    path = tmp_path / "large_non_ascii.csv"
    header = "地点,気温\n"
    rows = "\n".join(["富山,9.4", "高田,8.2"] * 5)
    _write_cp932(path, header + rows + "\n")

    df = csv_io.read_csv_auto(path)

    assert df.columns == ["地点", "気温"]
    assert set(df["地点"].to_list()) == {"富山", "高田"}


def test_read_csv_auto_parses_datetime_like_columns(tmp_path: Path) -> None:
    path = tmp_path / "datetime.csv"
    _write_utf8(path, "ts,value\n2020-01-01 00:00:00,1\n2020-01-02 00:00:00,2\n")
    df = csv_io.read_csv_auto(path)
    assert df.schema["ts"] == pl.Datetime
