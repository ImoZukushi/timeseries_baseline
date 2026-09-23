"""matplotlib図の共通ヘルパー。"""

from __future__ import annotations

import matplotlib.pyplot as plt


def ensure_japanese_font() -> None:
    """日本語フォントをmatplotlibに設定する。

    japanize-matplotlib は distutils 依存のためPython 3.12（setuptools未導入環境）では
    importできないので使用せず、Windowsに標準搭載されている日本語フォントを直接指定する。
    `sns.set_theme()` はフォント設定を含むrcParamsを上書きするため、各描画関数の先頭で
    都度呼び出して確実に日本語フォントが有効な状態で描画する。
    """
    plt.rcParams["font.family"] = "Meiryo"
    plt.rcParams["axes.unicode_minus"] = False  # Meiryoではマイナス記号が文字化けするため


def add_caption(fig: plt.Figure, text: str) -> None:
    """図の下部にキャプション（対象の説明文）を追加する。

    Args:
        fig: 対象のFigure。
        text: キャプション文字列。
    """
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=9, wrap=True)
