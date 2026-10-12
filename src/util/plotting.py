"""matplotlib図の共通ヘルパー。"""

from __future__ import annotations

import warnings
from functools import lru_cache

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font

# 既定の日本語フォント（Windows 標準搭載）
DEFAULT_JAPANESE_FONT = "Meiryo"
# 既定のフォントが使えない場合に、この順で探す日本語フォント（Windows・macOS・Linux の代表的なもの）
FALLBACK_JAPANESE_FONTS = (
    "Yu Gothic",
    "BIZ UDGothic",
    "MS Gothic",
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "IPAexGothic",
    "IPAGothic",
    "TakaoGothic",
    "VL Gothic",
)
# 日本語を描画できるかの判定に使う文字（ひらがな・カタカナ・漢字）
_JAPANESE_SAMPLE = "あア日"


def _supports_japanese(font_path: str) -> bool:
    """フォントファイルが日本語の文字（ひらがな・カタカナ・漢字）の字形を持つか。"""
    try:
        font = FT2Font(font_path)
    except (OSError, RuntimeError):
        # 読めないフォントファイルは使えないものとして扱う
        return False
    return all(font.get_char_index(ord(ch)) != 0 for ch in _JAPANESE_SAMPLE)


def _installed_fonts() -> dict[str, str]:
    """matplotlib が認識しているフォント名 → フォントファイルのパス。"""
    return {entry.name: entry.fname for entry in font_manager.fontManager.ttflist}


@lru_cache(maxsize=8)
def find_japanese_font(preferred: str = DEFAULT_JAPANESE_FONT) -> str | None:
    """使える日本語フォントを探す。

    次の順に、インストールされていて日本語の字形を持つフォントを返す。

    1. `preferred`（既定は `DEFAULT_JAPANESE_FONT`）
    2. `FALLBACK_JAPANESE_FONTS` の順
    3. それ以外のインストール済みフォント（名前順）

    フォントの一覧と字形の確認には時間がかかるため、結果はキャッシュする。

    Args:
        preferred: 優先して使うフォント名。

    Returns:
        フォント名。日本語を描画できるフォントが無ければ None。
    """
    installed = _installed_fonts()
    candidates = [preferred, *FALLBACK_JAPANESE_FONTS]
    for name in candidates:
        if name in installed and _supports_japanese(installed[name]):
            return name
    for name in sorted(installed):
        if name not in candidates and _supports_japanese(installed[name]):
            return name
    return None


def ensure_japanese_font(preferred: str = DEFAULT_JAPANESE_FONT) -> str | None:
    """使える日本語フォントを確認して、matplotlib に設定する。

    `preferred`（既定は Meiryo）が使えなければ、使える日本語フォントに切り替えて警告を出す。
    日本語フォントが1つも無ければ、フォントは変えずに警告を出す（日本語は □ で表示される）。

    japanize-matplotlib は distutils 依存のためPython 3.12（setuptools未導入環境）では
    importできないので使用しない。`sns.set_theme()` はフォント設定を含むrcParamsを上書きするため、
    各描画関数の先頭で都度呼び出して確実に日本語フォントが有効な状態で描画する。

    Args:
        preferred: 優先して使うフォント名。

    Returns:
        設定したフォント名。日本語フォントが無ければ None。
    """
    font = find_japanese_font(preferred)
    if font is None:
        _warn_once(
            f"日本語を描画できるフォントが見つかりません（{preferred} も使えません）。"
            "日本語は □ で表示されます"
        )
        return None
    if font != preferred:
        _warn_once(f"日本語フォント {preferred} が使えないため、{font} を使います")
    plt.rcParams["font.family"] = font
    plt.rcParams["axes.unicode_minus"] = (
        False  # 日本語フォントではマイナス記号が文字化けすることがあるため
    )
    return font


@lru_cache(maxsize=16)
def _warn_once(message: str) -> None:
    """同じ警告を、描画のたびに繰り返さないよう1回だけ出す。"""
    warnings.warn(message, stacklevel=3)


def add_caption(fig: plt.Figure, text: str) -> None:
    """図の下部にキャプション（対象の説明文）を追加する。

    Args:
        fig: 対象のFigure。
        text: キャプション文字列。
    """
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=9, wrap=True)
