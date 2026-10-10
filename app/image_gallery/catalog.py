"""画像ファイルの探索・絞り込み・並べ替え・ページ分割。"""

from __future__ import annotations

import datetime as dt
import fnmatch
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# 表示できる画像の拡張子（小文字）
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".svg")

# 探索しないディレクトリ名（仮想環境・キャッシュ・バージョン管理）
DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".ipynb_checkpoints",
        "node_modules",
    }
)

SortKey = Literal["name", "modified", "size"]


@dataclass(frozen=True)
class ImageFile:
    """一覧に出す画像ファイル。

    Attributes:
        path: 絶対パス。
        name: 表示ディレクトリからの相対パス（サブディレクトリを含む。`/` 区切り）。
        size: ファイルサイズ（バイト）。
        modified: 更新日時。
    """

    path: Path
    name: str
    size: int
    modified: dt.datetime

    @property
    def suffix(self) -> str:
        """拡張子（小文字、例: `.png`）。"""
        return self.path.suffix.lower()


@dataclass(frozen=True)
class ImageDirectory:
    """画像を含むディレクトリ。

    Attributes:
        path: 絶対パス。
        label: 画面に出す名前（ルートの親からの相対パス）。
        n_images: 直下の画像の数（サブディレクトリは含まない）。
    """

    path: Path
    label: str
    n_images: int


def _is_image(name: str, suffixes: Sequence[str]) -> bool:
    return os.path.splitext(name)[1].lower() in suffixes


def find_image_directories(
    roots: Sequence[Path],
    *,
    suffixes: Sequence[str] = IMAGE_SUFFIXES,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
) -> list[ImageDirectory]:
    """ルート以下で、画像を直下に含むディレクトリを列挙する（ラベル順）。

    Args:
        roots: 探すディレクトリ（存在しないものは無視）。
        suffixes: 画像とみなす拡張子。
        excluded_dirs: 中に入らないディレクトリ名（`.venv` など）。

    Returns:
        画像を含むディレクトリ。ラベルはルートからの相対パス（ルート自身は `.`）。
    """
    excluded = set(excluded_dirs)
    found: dict[Path, ImageDirectory] = {}
    for root in roots:
        if not root.is_dir():
            continue
        root = root.resolve()
        for current, dirnames, filenames in os.walk(root):
            # 除外ディレクトリには入らない（os.walk は dirnames の変更を反映する）
            dirnames[:] = sorted(d for d in dirnames if d not in excluded)
            n_images = sum(_is_image(f, suffixes) for f in filenames)
            if n_images:
                path = Path(current)
                label = path.relative_to(root).as_posix()
                if len(roots) > 1:
                    label = f"{root.name}/{label}" if label != "." else root.name
                found[path] = ImageDirectory(path, label, n_images)
    return sorted(found.values(), key=lambda d: d.label)


def list_images(
    directory: Path,
    *,
    recursive: bool = False,
    suffixes: Sequence[str] = IMAGE_SUFFIXES,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
) -> list[ImageFile]:
    """ディレクトリ内の画像ファイルを列挙する（名前順）。

    Args:
        directory: 対象のディレクトリ。
        recursive: サブディレクトリの画像も含めるか。
        suffixes: 画像とみなす拡張子。
        excluded_dirs: 再帰時に中に入らないディレクトリ名。

    Returns:
        画像ファイル。`name` は `directory` からの相対パス。

    Raises:
        NotADirectoryError: `directory` がディレクトリでない場合。
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"ディレクトリではありません: {directory}")
    directory = directory.resolve()
    excluded = set(excluded_dirs)
    images = []
    for current, dirnames, filenames in os.walk(directory):
        dirnames[:] = sorted(d for d in dirnames if d not in excluded) if recursive else []
        for filename in filenames:
            if not _is_image(filename, suffixes):
                continue
            path = Path(current) / filename
            stat = path.stat()
            images.append(
                ImageFile(
                    path=path,
                    name=path.relative_to(directory).as_posix(),
                    size=stat.st_size,
                    modified=dt.datetime.fromtimestamp(stat.st_mtime),
                )
            )
    return sorted(images, key=lambda f: f.name)


def filter_images(
    images: Sequence[ImageFile], query: str = "", suffixes: Sequence[str] | None = None
) -> list[ImageFile]:
    """名前と拡張子で絞り込む。

    Args:
        images: 画像ファイル。
        query: 名前（相対パス）の条件。`*` や `?` を含めばワイルドカード（例: `*__qq_*.png`）、
            含まなければ部分一致。大文字・小文字は区別しない。空なら絞り込まない。
        suffixes: 残す拡張子（Noneならすべて）。

    Returns:
        条件に合う画像ファイル（元の順序のまま）。
    """
    pattern = query.strip().lower()
    wildcard = any(c in pattern for c in "*?[")
    allowed = None if suffixes is None else {s.lower() for s in suffixes}

    def matches(image: ImageFile) -> bool:
        if allowed is not None and image.suffix not in allowed:
            return False
        if not pattern:
            return True
        name = image.name.lower()
        return fnmatch.fnmatch(name, pattern) if wildcard else pattern in name

    return [image for image in images if matches(image)]


def sort_images(
    images: Sequence[ImageFile], key: SortKey = "name", descending: bool = False
) -> list[ImageFile]:
    """名前・更新日時・サイズで並べ替える（同じ値なら名前順）。"""
    by_name = sorted(images, key=lambda f: f.name)
    if key == "name":
        return list(reversed(by_name)) if descending else by_name
    attr = {"modified": lambda f: f.modified, "size": lambda f: f.size}[key]
    return sorted(by_name, key=attr, reverse=descending)


def paginate[T](items: Sequence[T], page_size: int, page: int) -> tuple[list[T], int]:
    """1始まりのページ番号でページ分割する。

    Args:
        items: 全件。
        page_size: 1ページの件数（1以上）。
        page: ページ番号（範囲外なら最初・最後のページに丸める）。

    Returns:
        (そのページの要素, 総ページ数（0件でも1）)。

    Raises:
        ValueError: `page_size` が1未満の場合。
    """
    if page_size < 1:
        raise ValueError(f"page_size は1以上で指定してください: {page_size}")
    n_pages = max(1, math.ceil(len(items) / page_size))
    page = min(max(page, 1), n_pages)
    start = (page - 1) * page_size
    return list(items[start : start + page_size]), n_pages


def format_size(n_bytes: int) -> str:
    """ファイルサイズを読みやすい単位で表す（例: `12.3 KB`）。"""
    size = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
