"""matplotlib図の共通ヘルパー。"""

from __future__ import annotations

import contextlib
import functools
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 図の日本語フォント（プロジェクトの標準）。依存パッケージ matplotlib-fontja に同梱されている
JAPANESE_FONT = "IPAexGothic"
# IPAexGothic が使えない環境で代わりに使う日本語フォント（見つかった最初のもの）
FALLBACK_JAPANESE_FONTS = (
    "Meiryo",
    "Yu Gothic",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "Hiragino Sans",
    "IPAGothic",
)


@functools.cache
def _resolve_japanese_font() -> str:
    """使う日本語フォントの名前を決める（最初の1回だけ探す）。

    matplotlib-fontja を import すると、同梱の IPAexGothic（ipaexg.ttf）が matplotlib に登録される。
    登録できない・見つからない場合は警告を出し、`FALLBACK_JAPANESE_FONTS` の最初に
    見つかったものを使う。
    """
    # import 時に同梱の IPAexGothic を登録する（入っていなければ代わりのフォントを探す）
    with contextlib.suppress(ImportError):
        import matplotlib_fontja  # noqa: F401
    available = {font.name for font in font_manager.fontManager.ttflist}
    if JAPANESE_FONT in available:
        return JAPANESE_FONT
    fallback = next((name for name in FALLBACK_JAPANESE_FONTS if name in available), None)
    if fallback is None:
        warnings.warn(
            f"日本語フォント {JAPANESE_FONT} も代わりのフォントも見つかりません。"
            "図の日本語が表示されない可能性があります"
            "（uv sync で matplotlib-fontja を入れてください）",
            stacklevel=3,
        )
        return str(plt.rcParams["font.family"][0])
    warnings.warn(
        f"日本語フォント {JAPANESE_FONT} が見つからないため、{fallback} を使います"
        "（uv sync で matplotlib-fontja を入れてください）",
        stacklevel=3,
    )
    return fallback


def _with_font_first(font: str, names: list[str]) -> list[str]:
    """フォント名のリストの先頭に font を置き、DejaVu 系のフォントを外す。"""
    return [font, *[name for name in names if name != font and not name.startswith("DejaVu")]]


def ensure_japanese_font() -> str:
    """日本語フォント（標準は IPAexGothic）をmatplotlibに設定し、DejaVu Sans を使わないようにする。

    通常の文字に加え、総称名（sans-serif など）で指定された文字・数式（$...$）も
    日本語フォントで描く。

    IPAexGothic は依存パッケージ matplotlib-fontja に同梱されたものを使うため、
    OS に関係なく同じフォントで描画される。見つからない環境では警告を出し、
    手元の日本語フォント（Meiryo など）を使う。
    `sns.set_theme()` はフォント設定を含むrcParamsを上書きするため、各描画関数の先頭で
    都度呼び出して確実に日本語フォントが有効な状態で描画する。

    Returns:
        設定したフォントの名前。
    """
    font = _resolve_japanese_font()
    # 総称名（sans-serif・monospace など）で指定された文字も日本語フォントで描く。
    # matplotlib の既定では総称名の先頭が DejaVu 系なので、DejaVu 系はリストから外す
    generic_families = {
        "sans-serif": _with_font_first(font, plt.rcParams["font.sans-serif"]),
        "serif": _with_font_first(font, plt.rcParams["font.serif"]),
        "monospace": _with_font_first(font, plt.rcParams["font.monospace"]),
        "cursive": _with_font_first(font, plt.rcParams["font.cursive"]),
        "fantasy": _with_font_first(font, plt.rcParams["font.fantasy"]),
    }
    matplotlib.rc("font", family=font, **generic_families)
    # 数式（$...$。指数の表記など）も日本語フォントで描く（既定の fontset は DejaVu Sans）。
    # IPAexGothic には太字が無いため、数式の太字も通常の太さで描く
    # （太字を探して警告が出ないように）。
    # 日本語フォントに無い数学記号は STIX で描く
    matplotlib.rc(
        "mathtext",
        fontset="custom",
        rm=font,
        sf=font,
        tt=font,
        cal=font,
        bf=font,
        it=f"{font}:italic",
        bfit=f"{font}:italic",
        fallback="stix",
    )
    # マイナス記号を ASCII のハイフンで描く（日本語フォントによってはU+2212が化けるため）
    matplotlib.rc("axes", unicode_minus=False)
    return font


def add_caption(fig: plt.Figure, text: str) -> None:
    """図の下部にキャプション（対象の説明文）を追加する。

    Args:
        fig: 対象のFigure。
        text: キャプション文字列。
    """
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=9, wrap=True)
