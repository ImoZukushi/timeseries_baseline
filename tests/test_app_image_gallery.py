"""app/image_gallery（画像ギャラリー）のテスト。"""

from __future__ import annotations

import datetime as dt
import io
import os
from pathlib import Path

import pytest
from image_gallery.catalog import (
    ImageFile,
    filter_images,
    find_image_directories,
    format_size,
    list_images,
    paginate,
    sort_images,
)
from image_gallery.thumbnails import image_info, make_thumbnail
from PIL import Image

MAIN = Path(__file__).resolve().parents[1] / "app" / "image_gallery" / "main.py"


def _save_image(path: Path, size: tuple[int, int] = (40, 20), mode: str = "RGB") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color=0).save(path)
    return path


@pytest.fixture
def gallery(tmp_path: Path) -> Path:
    """画像のあるディレクトリ構成（除外ディレクトリ・画像以外のファイルを含む）。"""
    root = tmp_path / "repo"
    _save_image(root / "figures" / "a__qq_plot.png")
    _save_image(root / "figures" / "b__acf.jpg")
    _save_image(root / "figures" / "sub" / "c__qq_plot.png")
    _save_image(root / ".venv" / "lib" / "icon.png")
    (root / "figures" / "notes.txt").write_text("x", encoding="utf-8")
    (root / "figures" / "d.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>', encoding="utf-8"
    )
    (root / "empty").mkdir()
    return root


def _image(name: str, size: int, day: int) -> ImageFile:
    return ImageFile(Path(name), name, size, dt.datetime(2026, 1, day))


# --- 探索 ------------------------------------------------------------------------------------


def test_find_image_directories_skips_excluded(gallery: Path) -> None:
    found = {d.label: d.n_images for d in find_image_directories([gallery])}
    # .venv の中は探さず、画像の無いディレクトリは出さない。数は直下の画像のみ
    assert found == {"figures": 3, "figures/sub": 1}


def test_find_image_directories_multiple_roots_prefix_root_name(
    gallery: Path, tmp_path: Path
) -> None:
    other = tmp_path / "other"
    _save_image(other / "x.png")
    labels = [d.label for d in find_image_directories([gallery, other, tmp_path / "missing"])]
    assert labels == ["other", "repo/figures", "repo/figures/sub"]


def test_list_images_recursive(gallery: Path) -> None:
    flat = list_images(gallery / "figures")
    assert [f.name for f in flat] == ["a__qq_plot.png", "b__acf.jpg", "d.svg"]
    deep = list_images(gallery / "figures", recursive=True)
    assert [f.name for f in deep] == ["a__qq_plot.png", "b__acf.jpg", "d.svg", "sub/c__qq_plot.png"]
    assert all(f.size > 0 for f in deep)
    with pytest.raises(NotADirectoryError):
        list_images(gallery / "figures" / "notes.txt")


# --- 絞り込み・並べ替え・ページ ----------------------------------------------------------------


def test_filter_images_substring_wildcard_and_suffix(gallery: Path) -> None:
    images = list_images(gallery / "figures", recursive=True)
    names = lambda xs: [x.name for x in xs]  # noqa: E731
    assert names(filter_images(images, "QQ")) == ["a__qq_plot.png", "sub/c__qq_plot.png"]
    # ワイルドカードは相対パス全体に一致させる
    assert names(filter_images(images, "*/c__*")) == ["sub/c__qq_plot.png"]
    assert names(filter_images(images, "", [".jpg", ".svg"])) == ["b__acf.jpg", "d.svg"]
    assert names(filter_images(images, "qq", [".jpg"])) == []
    assert len(filter_images(images)) == 4


def test_sort_images() -> None:
    images = [_image("b.png", 30, 1), _image("a.png", 10, 3), _image("c.png", 10, 2)]
    names = lambda xs: [x.name for x in xs]  # noqa: E731
    assert names(sort_images(images)) == ["a.png", "b.png", "c.png"]
    assert names(sort_images(images, "name", descending=True)) == ["c.png", "b.png", "a.png"]
    assert names(sort_images(images, "modified", descending=True)) == ["a.png", "c.png", "b.png"]
    # 同じサイズなら名前順
    assert names(sort_images(images, "size")) == ["a.png", "c.png", "b.png"]


def test_paginate() -> None:
    items = list(range(25))
    assert paginate(items, 10, 1) == (list(range(10)), 3)
    assert paginate(items, 10, 3) == ([20, 21, 22, 23, 24], 3)
    # 範囲外のページは端に丸める
    assert paginate(items, 10, 99)[0] == [20, 21, 22, 23, 24]
    assert paginate(items, 10, 0)[0] == list(range(10))
    assert paginate([], 10, 1) == ([], 1)
    with pytest.raises(ValueError):
        paginate(items, 0, 1)


def test_format_size() -> None:
    assert format_size(512) == "512 B"
    assert format_size(2048) == "2.0 KB"
    assert format_size(5 * 1024**2) == "5.0 MB"


# --- サムネイル・情報 -------------------------------------------------------------------------


def test_make_thumbnail_keeps_aspect_ratio(tmp_path: Path) -> None:
    path = _save_image(tmp_path / "big.png", size=(1200, 600), mode="P")
    data = make_thumbnail(path, max_size=300)
    assert data is not None
    with Image.open(io.BytesIO(data)) as thumb:
        assert thumb.format == "PNG"
        assert thumb.size == (300, 150)
    # 小さい画像は拡大しない
    small = make_thumbnail(_save_image(tmp_path / "small.png", size=(40, 20)), max_size=300)
    assert small is not None
    with Image.open(io.BytesIO(small)) as thumb:
        assert thumb.size == (40, 20)
    with pytest.raises(ValueError):
        make_thumbnail(path, max_size=0)


def test_thumbnail_and_info_for_svg_and_broken_files(gallery: Path, tmp_path: Path) -> None:
    svg = gallery / "figures" / "d.svg"
    assert make_thumbnail(svg) is None
    assert image_info(svg).format == "SVG"
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    assert make_thumbnail(broken) is None
    info = image_info(broken)
    assert info.width is None and info.error


def test_image_info(gallery: Path) -> None:
    info = image_info(gallery / "figures" / "b__acf.jpg")
    assert (info.width, info.height, info.format) == (40, 20, "JPEG")


# --- 画面 ------------------------------------------------------------------------------------


def test_app_lists_images(gallery: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("IMAGE_GALLERY_ROOTS", str(gallery))
    at = AppTest.from_file(str(MAIN), default_timeout=60).run()
    assert not at.exception
    # 最初のディレクトリ（figures）の直下3枚が並ぶ
    assert len(at.get("image")) == 3
    assert any("3 / 3 枚" in c.value for c in at.caption)
    # サブディレクトリを含め、名前で絞り込む
    at.sidebar.checkbox[0].check().run()
    at.sidebar.text_input[0].input("qq").run()
    assert not at.exception
    assert len(at.get("image")) == 2
    assert any("2 / 4 枚" in c.value for c in at.caption)


def test_app_without_images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("IMAGE_GALLERY_ROOTS", str(tmp_path) + os.pathsep)
    at = AppTest.from_file(str(MAIN), default_timeout=60).run()
    assert not at.exception
    assert "見つかりません" in at.info[0].value
