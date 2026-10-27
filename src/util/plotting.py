"""matplotlib図の共通ヘルパー。"""

from __future__ import annotations

import functools
import importlib.util
import logging
import os
import warnings
from collections.abc import Iterator
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

from util.paths import get_repo_root

# 図の日本語フォント（プロジェクトの標準）。依存パッケージ matplotlib-fontja に同梱されている
JAPANESE_FONT = "IPAexGothic"
JAPANESE_FONT_FILE = "ipaexg.ttf"
# IPAexGothic のフォントファイルのパスを指定する環境変数（自動で見つからない環境用）
FONT_PATH_ENV = "MPL_JAPANESE_FONT_PATH"
# IPAexGothic のフォントファイルを同梱しているパッケージ（import はせず、ファイルだけを使う）
FONT_PACKAGES = ("matplotlib_fontja", "japanize_matplotlib")
# IPAexGothic が使えない環境で代わりに使う日本語フォント（見つかった最初のもの）。
# Windows・macOS・Linux（SageMaker Studio などのコンテナを含む）の代表的なもの
FALLBACK_JAPANESE_FONTS = (
    "Meiryo",
    "Yu Gothic",
    "BIZ UDGothic",
    "Hiragino Sans",
    "Noto Sans CJK JP",
    "Noto Sans JP",
    "Source Han Sans JP",
    "IPAGothic",
    "IPAPGothic",
    "TakaoGothic",
    "TakaoPGothic",
    "VL Gothic",
    "MS Gothic",
)
# 日本語フォントが無いときに出続ける警告（最初に1回だけ説明の警告を出し、以降は出さない）
_MISSING_GLYPH_WARNING = r"Glyph \d+ .* missing from font"


def _font_file_candidates() -> Iterator[Path]:
    """IPAexGothic のフォントファイルの候補を、優先する順に返す（存在は確かめない）。

    1. 環境変数 `MPL_JAPANESE_FONT_PATH` で指定されたパス
    2. matplotlib-fontja・japanize-matplotlib の配布物の中（import しないので、Python 3.12 で
       import できない japanize-matplotlib でも使える）
    3. リポジトリの `.venv`（Jupyter のカーネルが別の Python 環境でも、uv sync 済みなら使える）
    """
    if env_path := os.environ.get(FONT_PATH_ENV):
        yield Path(env_path)
    for package in FONT_PACKAGES:
        try:
            spec = importlib.util.find_spec(package)
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin is not None:
            yield Path(spec.origin).parent / "fonts" / JAPANESE_FONT_FILE
    venv = get_repo_root() / ".venv"
    for pattern in ("Lib/site-packages", "lib/python*/site-packages"):
        for site_packages in sorted(venv.glob(pattern)):
            yield site_packages / FONT_PACKAGES[0] / "fonts" / JAPANESE_FONT_FILE


def _registered_fonts() -> set[str]:
    """matplotlib に登録されているフォントの名前。"""
    return {font.name for font in font_manager.fontManager.ttflist}


def _register_japanese_font_file() -> bool:
    """IPAexGothic のフォントファイルを探して matplotlib に登録する。登録できれば True。

    OS にインストールされていて matplotlib のフォント一覧（キャッシュ）にまだ無い場合も、
    OS のフォントのフォルダから探して登録する。
    """
    candidates = list(_font_file_candidates())
    # OS のフォント（apt の fonts-ipaexfont・~/.fonts など）。キャッシュの後に入れた場合にも見つける
    candidates += [
        Path(path)
        for path in font_manager.findSystemFonts()
        if Path(path).name.lower() == JAPANESE_FONT_FILE
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            font_manager.fontManager.addfont(path)
        except (OSError, RuntimeError, ValueError):
            continue
        if JAPANESE_FONT in _registered_fonts():
            return True
    return False


def _register_system_fonts() -> None:
    """OS のフォントのうち、matplotlib にまだ登録されていないものを登録する。"""
    registered = {Path(font.fname) for font in font_manager.fontManager.ttflist}
    for path in font_manager.findSystemFonts():
        if Path(path) in registered:
            continue
        try:
            font_manager.fontManager.addfont(path)
        except (OSError, RuntimeError, ValueError):
            # 読めないフォントファイルは飛ばす
            continue


@functools.cache
def _resolve_japanese_font() -> str | None:
    """使う日本語フォントの名前を決める（最初の1回だけ探す）。

    1. IPAexGothic（登録済みか、フォントファイルが見つかれば登録して使う）
    2. 見つからなければ `FALLBACK_JAPANESE_FONTS` の最初に見つかったもの（警告を1回出す）。
       フォント一覧のキャッシュが古い可能性があるので、見つからなければ1回だけ作り直して探す
    3. どれも無ければ None（警告を1回出す）

    Returns:
        フォント名。日本語フォントが無ければ None。
    """
    if JAPANESE_FONT in _registered_fonts() or _register_japanese_font_file():
        return JAPANESE_FONT
    available = _registered_fonts()
    fallback = next((name for name in FALLBACK_JAPANESE_FONTS if name in available), None)
    if fallback is None:
        # フォント一覧のキャッシュを作った後にフォントを入れた環境のため、まだ登録されていない
        # OS のフォントを登録してから探し直す（fontManager は他のモジュールも参照しているので
        # 作り直さずに追加する）
        _register_system_fonts()
        available = _registered_fonts()
        fallback = next((name for name in FALLBACK_JAPANESE_FONTS if name in available), None)
    hint = (
        "matplotlib-fontja を入れる"
        "（uv sync、または Jupyter のカーネルで pip install matplotlib-fontja）"
        f"か、環境変数 {FONT_PATH_ENV} に {JAPANESE_FONT_FILE} のパスを指定してください"
    )
    if fallback is None:
        warnings.warn(
            f"日本語フォント（{JAPANESE_FONT} とその代わり）が見つかりません。"
            f"図の日本語は表示されません（□になります）。{hint}",
            stacklevel=3,
        )
        return None
    warnings.warn(
        f"日本語フォント {JAPANESE_FONT} が見つからないため、{fallback} を使います。{hint}",
        stacklevel=3,
    )
    return fallback


class _MissingGlyphLogFilter(logging.Filter):
    """数式（mathtext）の描画で出る「フォントにこの文字が無い」ログを出さないフィルタ。"""

    def filter(self, record: logging.LogRecord) -> bool:
        return "does not have a glyph" not in record.getMessage()


@functools.cache
def _install_missing_glyph_log_filter() -> None:
    """mathtext のロガーに `_MissingGlyphLogFilter` を付ける（1回だけ）。"""
    logging.getLogger("matplotlib.mathtext").addFilter(_MissingGlyphLogFilter())


def _with_font_first(font: str, names: list[str]) -> list[str]:
    """フォント名のリストの先頭に font を置き、DejaVu 系のフォントを外す。"""
    return [font, *[name for name in names if name != font and not name.startswith("DejaVu")]]


def ensure_japanese_font() -> str:
    """日本語フォント（標準は IPAexGothic）をmatplotlibに設定し、DejaVu Sans を使わないようにする。

    通常の文字に加え、総称名（sans-serif など）で指定された文字・数式（$...$）も
    日本語フォントで描く。

    IPAexGothic は依存パッケージ matplotlib-fontja に同梱されたフォントファイルを、import せずに
    探して登録する（環境変数 `MPL_JAPANESE_FONT_PATH`・リポジトリの `.venv`・OS のフォントも探す）。
    そのため SageMaker Studio などの Linux のコンテナでも、OS に関係なく同じフォントで描画される。
    見つからない環境では警告を1回だけ出し、手元の日本語フォント（Noto Sans CJK JP など）を使う。
    日本語フォントが1つも無い場合は、フォントの設定を変えず、文字が無いという警告も出し続けない
    （学習中に図を描くたびに警告が出ないように）。
    `sns.set_theme()` はフォント設定を含むrcParamsを上書きするため、各描画関数の先頭で
    都度呼び出して確実に日本語フォントが有効な状態で描画する。

    Returns:
        設定したフォントの名前（日本語フォントが無ければ、既定のフォントの名前）。
    """
    font = _resolve_japanese_font()
    # マイナス記号を ASCII のハイフンで描く（日本語フォントによってはU+2212が化けるため）
    matplotlib.rc("axes", unicode_minus=False)
    if font is None:
        # 日本語フォントが無い: 既定のフォントのまま描き、図ごとに出る「文字が無い」警告は出さない
        # （説明の警告は最初に1回だけ出している）
        warnings.filterwarnings("ignore", message=_MISSING_GLYPH_WARNING, category=UserWarning)
        _install_missing_glyph_log_filter()
        return font_manager.FontProperties(family=plt.rcParams["font.family"]).get_name()
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
    return font


def add_caption(fig: plt.Figure, text: str) -> None:
    """図の下部にキャプション（対象の説明文）を追加する。

    Args:
        fig: 対象のFigure。
        text: キャプション文字列。
    """
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=9, wrap=True)
