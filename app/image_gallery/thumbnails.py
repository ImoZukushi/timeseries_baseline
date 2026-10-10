"""サムネイルの作成と画像の情報の取得（Pillow）。

一覧に元の画像をそのまま並べると、大きな図（数千px）が多いときに表示が重くなるため、
縮小した PNG のバイト列を作って表示する。SVG はベクター形式で縮小の必要がないため対象外。
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

# Pillow で開けない（ラスターではない）形式
VECTOR_SUFFIXES = (".svg",)


@dataclass(frozen=True)
class ImageInfo:
    """画像の情報。

    Attributes:
        width: 幅（px）。SVG・読めない画像はNone。
        height: 高さ（px）。
        format: 形式（`PNG` など）。
        mode: 色の形式（`RGBA` など）。
        n_frames: フレーム数（アニメーションGIFなど）。
        error: 読み込めなかった場合の理由。
    """

    width: int | None = None
    height: int | None = None
    format: str | None = None
    mode: str | None = None
    n_frames: int = 1
    error: str | None = None


def image_info(path: Path) -> ImageInfo:
    """画像の画素数・形式を返す（画像全体は読み込まない）。読めない場合は `error` に理由。"""
    if path.suffix.lower() in VECTOR_SUFFIXES:
        return ImageInfo(format="SVG")
    try:
        with Image.open(path) as img:
            return ImageInfo(
                width=img.width,
                height=img.height,
                format=img.format,
                mode=img.mode,
                n_frames=getattr(img, "n_frames", 1),
            )
    except (OSError, UnidentifiedImageError) as e:
        return ImageInfo(error=str(e))


def make_thumbnail(path: Path, max_size: int = 480) -> bytes | None:
    """長辺を `max_size` px 以下に縮小した PNG のバイト列を返す。

    EXIF の向き情報（スマートフォンの写真など）を反映し、縦横比は保つ。アニメーションは
    最初のフレームを使う。SVG・読めない画像は None（呼び出し側で元ファイルを表示する）。

    Raises:
        ValueError: `max_size` が1未満の場合。
    """
    if max_size < 1:
        raise ValueError(f"max_size は1以上で指定してください: {max_size}")
    if path.suffix.lower() in VECTOR_SUFFIXES:
        return None
    try:
        with Image.open(path) as img:
            thumb = ImageOps.exif_transpose(img)
            if thumb.mode not in ("RGB", "RGBA", "L", "LA"):
                thumb = thumb.convert("RGBA")
            thumb.thumbnail((max_size, max_size))
            buffer = io.BytesIO()
            thumb.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()
    except (OSError, UnidentifiedImageError):
        return None
