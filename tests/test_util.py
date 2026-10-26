"""util パッケージ（paths / plotting / csv_io）のテスト。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib import font_manager

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


def test_ensure_japanese_font_sets_ipaexgothic() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    # matplotlib-fontja に同梱の IPAexGothic が標準（OS にインストールされていなくても使える）
    assert plotting.ensure_japanese_font() == "IPAexGothic"
    assert plt.rcParams["font.family"] == ["IPAexGothic"]
    assert plt.rcParams["axes.unicode_minus"] is False
    # 実際に IPAexGothic のフォントファイルが使われる（見つからず別のフォントに置き換わらない）
    path = font_manager.findfont("IPAexGothic", fallback_to_default=False)
    assert Path(path).name == "ipaexg.ttf"


def test_ensure_japanese_font_does_not_use_dejavu(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    import numpy as np
    import seaborn as sns

    # 図を描くときに読み込まれるフォントファイルを記録する
    loaded: list[str] = []
    original = font_manager._get_font

    def recording_get_font(font_filepaths: tuple, *args: object, **kwargs: object) -> object:
        # 要素は (フォントファイルのパス, フェイス番号) の組
        loaded.extend(Path(str(p[0] if isinstance(p, tuple) else p)).name for p in font_filepaths)
        return original(font_filepaths, *args, **kwargs)

    monkeypatch.setattr(font_manager, "_get_font", recording_get_font)
    # seaborn のテーマ（フォントを既定に戻す）の後に呼ぶ、いつもの使い方
    sns.set_theme(style="whitegrid")
    plotting.ensure_japanese_font()
    for key in ("font.sans-serif", "font.serif", "font.monospace"):
        assert not [f for f in plt.rcParams[key] if f.startswith("DejaVu")], key
    fig, ax = plt.subplots()
    try:
        # 日本語・数式・指数の表記・マイナス・凡例を含む図
        ax.plot(np.arange(5) * 1e6, [-1.0, 2.0, -3.0, 4.0, 5.0], label="系列①")
        ax.set_title(r"日本語の見出し $\alpha^2 + \mathbf{x}$")
        ax.text(0, 0, "monospace の文字", family="monospace")
        ax.legend()
        fig.savefig(io.BytesIO(), format="png")
    finally:
        plt.close(fig)
    assert "ipaexg.ttf" in loaded
    assert not [name for name in loaded if name.startswith("DejaVu")], sorted(set(loaded))


def test_ensure_japanese_font_falls_back_with_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    plotting._resolve_japanese_font.cache_clear()
    monkeypatch.setattr(plotting, "JAPANESE_FONT", "存在しないフォント")
    monkeypatch.setattr(plotting, "FALLBACK_JAPANESE_FONTS", ("IPAexGothic",))
    try:
        with pytest.warns(UserWarning, match="存在しないフォント"):
            assert plotting.ensure_japanese_font() == "IPAexGothic"
    finally:
        # 他のテストに影響しないよう、探した結果を消しておく
        plotting._resolve_japanese_font.cache_clear()


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
    assert list(df.columns) == ["地点", "気温"]
    assert df["地点"].tolist() == ["富山", "高田"]


def test_read_csv_auto_large_ascii_data_skips_full_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # 大容量ファイル判定の閾値を下げて、全文デコードしない経路を強制する。
    monkeypatch.setattr(csv_io, "LARGE_FILE_THRESHOLD_BYTES", 10)
    path = tmp_path / "large_ascii.csv"
    header = "年月日時,風速(瞬時),風向(瞬時)\n"
    rows = "\n".join(f"2020-01-01 00:00:{i:02d},{i * 0.1:.1f},{i}" for i in range(100))
    _write_cp932(path, header + rows + "\n")

    df = csv_io.read_csv_auto(path)

    assert list(df.columns) == ["年月日時", "風速(瞬時)", "風向(瞬時)"]
    assert len(df) == 100
    assert df["風向(瞬時)"].tolist()[:3] == [0, 1, 2]


def test_read_csv_auto_large_non_ascii_data_falls_back_to_full_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(csv_io, "LARGE_FILE_THRESHOLD_BYTES", 10)
    path = tmp_path / "large_non_ascii.csv"
    header = "地点,気温\n"
    rows = "\n".join(["富山,9.4", "高田,8.2"] * 5)
    _write_cp932(path, header + rows + "\n")

    df = csv_io.read_csv_auto(path)

    assert list(df.columns) == ["地点", "気温"]
    assert set(df["地点"].tolist()) == {"富山", "高田"}


def test_read_csv_auto_parses_datetime_like_columns(tmp_path: Path) -> None:
    path = tmp_path / "datetime.csv"
    _write_utf8(path, "ts,value\n2020-01-01 00:00:00,1\n2020-01-02 00:00:00,2\n")
    df = csv_io.read_csv_auto(path)
    assert pd.api.types.is_datetime64_any_dtype(df["ts"])


def test_read_csv_auto_parses_iso_datetime(tmp_path: Path) -> None:
    # ISO 8601 の `2016-01-19T00:00:00.000000` 形式（小数秒なしも）を日時として読む
    path = tmp_path / "iso.csv"
    _write_utf8(path, "ts,v\n2016-01-19T00:00:00.000000,1\n2016-01-20T12:30:15.000000,2\n")
    df = csv_io.read_csv_auto(path)
    assert pd.api.types.is_datetime64_any_dtype(df["ts"])
    assert df["ts"].tolist() == [
        pd.Timestamp(dt.datetime(2016, 1, 19)),
        pd.Timestamp(dt.datetime(2016, 1, 20, 12, 30, 15)),
    ]
    _write_utf8(path, "ts,v\n2016-01-19T00:00:00,1\n")
    assert pd.api.types.is_datetime64_any_dtype(csv_io.read_csv_auto(path)["ts"])


def test_read_csv_auto_null_markers(tmp_path: Path) -> None:
    # "*" と空文字は欠損、"NA" などは値として残す
    path = tmp_path / "null.csv"
    _write_utf8(path, "a,b\n1,x\n*,NA\n3,\n")
    df = csv_io.read_csv_auto(path)
    assert df["a"].isna().tolist() == [False, True, False]
    assert df["b"].tolist()[:2] == ["x", "NA"] and pd.isna(df["b"].iloc[2])


def test_iter_csv_chunks_keeps_dtypes_across_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(csv_io, "LARGE_FILE_THRESHOLD_BYTES", 10)
    path = tmp_path / "large.csv"
    header = "年月日時,風速(瞬時),地点\n"
    # 後半の部分にだけ欠測記号 "*" がある（部分ごとに型がぶれないことを確かめる）
    rows = [f"2020-01-01 00:{i // 60:02d}:{i % 60:02d},{i},A" for i in range(50)]
    rows += [f"2020-01-01 01:{i // 60:02d}:{i % 60:02d},*,B" for i in range(50)]
    _write_cp932(path, header + "\n".join(rows) + "\n")
    chunks = list(csv_io.iter_csv_chunks(path, chunksize=30, datetime_sample_rows=20))
    assert len(chunks) == 4
    assert sum(len(c) for c in chunks) == 100
    for chunk in chunks:
        assert pd.api.types.is_datetime64_any_dtype(chunk["年月日時"])
        assert chunk["風速(瞬時)"].dtype == "float64"
    assert chunks[-1]["風速(瞬時)"].isna().all()


def test_iter_csv_chunks_small_file_returns_one_frame(tmp_path: Path) -> None:
    path = tmp_path / "small.csv"
    _write_utf8(path, "ts,v\n2020-01-01,1\n")
    chunks = list(csv_io.iter_csv_chunks(path))
    assert len(chunks) == 1 and len(chunks[0]) == 1
