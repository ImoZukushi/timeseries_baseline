"""util パッケージ（paths / plotting / csv_io）のテスト。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import japanize_matplotlib  # noqa: F401  # 日本語フォント（IPAexGothic）を matplotlib に登録する
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


@pytest.fixture
def fake_fonts(monkeypatch: pytest.MonkeyPatch) -> dict[str, bool]:
    """インストール済みフォントを差し替える（フォント名 → 日本語の字形を持つか）。"""
    fonts: dict[str, bool] = {}
    monkeypatch.setattr(plotting, "_installed_fonts", lambda: {name: name for name in fonts})
    monkeypatch.setattr(plotting, "_supports_japanese", lambda path: fonts[path])
    # 探索結果・警告のキャッシュを、テストの前後で空にする
    plotting.find_japanese_font.cache_clear()
    plotting._warn_once.cache_clear()
    yield fonts
    plotting.find_japanese_font.cache_clear()
    plotting._warn_once.cache_clear()


def test_ensure_japanese_font_uses_default_when_available(fake_fonts: dict[str, bool]) -> None:
    fake_fonts |= {"DejaVu Sans": False, "Meiryo": True, "Yu Gothic": True}
    plt.rcParams["font.family"] = "DejaVu Sans"
    assert plotting.ensure_japanese_font() == "Meiryo"
    assert plt.rcParams["font.family"] == ["Meiryo"]
    assert plt.rcParams["axes.unicode_minus"] is False


def test_ensure_japanese_font_falls_back_with_warning(fake_fonts: dict[str, bool]) -> None:
    # 既定の Meiryo が無い環境では、候補の順（Yu Gothic が MS Gothic より先）に切り替える
    fake_fonts |= {"DejaVu Sans": False, "MS Gothic": True, "Yu Gothic": True}
    with pytest.warns(UserWarning, match="Meiryo が使えないため、Yu Gothic"):
        assert plotting.ensure_japanese_font() == "Yu Gothic"
    assert plt.rcParams["font.family"] == ["Yu Gothic"]


def test_ensure_japanese_font_skips_fonts_without_japanese_glyphs(
    fake_fonts: dict[str, bool],
) -> None:
    # 名前があっても日本語の字形が無いフォントは使わず、候補外のフォントも探す
    fake_fonts |= {"Meiryo": False, "Some JP Font": True}
    with pytest.warns(UserWarning, match="Some JP Font"):
        assert plotting.ensure_japanese_font() == "Some JP Font"


def test_ensure_japanese_font_keeps_font_when_none_available(fake_fonts: dict[str, bool]) -> None:
    fake_fonts |= {"DejaVu Sans": False}
    plt.rcParams["font.family"] = "DejaVu Sans"
    with pytest.warns(UserWarning, match="見つかりません"):
        assert plotting.ensure_japanese_font() is None
    assert plt.rcParams["font.family"] == ["DejaVu Sans"]


def test_japanize_matplotlib_registers_ipaexgothic() -> None:
    # japanize-matplotlib を import すると、同梱の IPAexGothic が常に使える（日本語フォントの最後の頼み）
    installed = plotting._installed_fonts()
    assert "IPAexGothic" in installed
    assert plotting._supports_japanese(installed["IPAexGothic"])
    plotting.find_japanese_font.cache_clear()
    assert plotting.find_japanese_font("No Such Font") is not None
    plotting.find_japanese_font.cache_clear()


def test_find_japanese_font_on_this_machine() -> None:
    # 実際の環境でも、見つかったフォントは日本語の字形を持つ
    plotting.find_japanese_font.cache_clear()
    font = plotting.find_japanese_font()
    if font is not None:
        path = plotting._installed_fonts()[font]
        assert plotting._supports_japanese(path)


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


def test_read_csv_auto_parses_iso_datetime_written_by_polars(tmp_path: Path) -> None:
    # polars の write_csv は日時列を `2016-01-19T00:00:00.000000` の形で書き出す
    path = tmp_path / "iso.csv"
    original = pl.DataFrame(
        {"ts": [dt.datetime(2016, 1, 19), dt.datetime(2016, 1, 20, 12, 30, 15)], "v": [1, 2]}
    )
    original.write_csv(path)
    assert "T" in path.read_text(encoding="utf-8")
    df = csv_io.read_csv_auto(path)
    assert df.schema["ts"] == pl.Datetime
    assert df["ts"].to_list() == original["ts"].to_list()
