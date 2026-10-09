"""縦に並べた時系列の Plotly 図（全段をまたぐカーソル付き）。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from eda.time_series_eda import break_line_at_gaps
from timeseries_viewer.series import TIME, VALUE

ROW_HEIGHT = 230  # 1段あたりの高さ（px）
_LINE_COLORS = ("#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3")
_CURSOR_COLOR = "#d62728"


@dataclass
class SeriesView:
    """1段分の描画内容。

    Attributes:
        label: 段の見出し。
        data: 列 `time`, `value` の DataFrame（時刻順）。
        y_range: 縦軸の範囲（Noneなら自動）。
        log_y: 縦軸を対数にするか。
        cursor_point: カーソル時刻に最も近い観測の (時刻, 値)。
    """

    label: str
    data: pl.DataFrame
    y_range: tuple[float, float] | None = None
    log_y: bool = False
    cursor_point: tuple[Any, float] | None = None


def _line_xy(data: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """折れ線の座標（時刻の大きな空白では線を切る）。"""
    x = data[TIME].to_numpy()
    y = data[VALUE].to_numpy().astype(np.float64)
    return x, break_line_at_gaps(x, y)


def build_figure(
    views: Sequence[SeriesView],
    *,
    x_range: tuple[Any, Any] | None = None,
    cursor: Any = None,
) -> go.Figure:
    """系列を縦に並べた図を作る（横軸は全段で共有）。

    Args:
        views: 段ごとの描画内容（上から順）。
        x_range: 横軸の表示範囲（Noneなら自動）。
        cursor: カーソルの時刻。全段をまたぐ縦線と、各段の最も近い観測の点を描く。

    Returns:
        Plotly の Figure。
    """
    n = max(1, len(views))
    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=min(0.06, 0.3 / n),
        subplot_titles=[v.label for v in views] or None,
    )
    for row, view in enumerate(views, start=1):
        x, y = _line_xy(view.data)
        color = _LINE_COLORS[(row - 1) % len(_LINE_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                name=view.label,
                line={"color": color, "width": 1.2},
                # 点を小さく表示しておくと、線の近くのクリックでその点が選択される
                marker={"color": color, "size": 3},
                connectgaps=False,
                hovertemplate="%{y:.4g}<extra>" + view.label + "</extra>",
            ),
            row=row,
            col=1,
        )
        if view.cursor_point is not None:
            fig.add_trace(
                go.Scatter(
                    x=[view.cursor_point[0]],
                    y=[view.cursor_point[1]],
                    mode="markers",
                    marker={"color": _CURSOR_COLOR, "size": 10, "symbol": "circle-open-dot"},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=row,
                col=1,
            )
        _apply_y_axis(fig, row, view)

    if cursor is not None:
        # 横軸は全段で共有なので、紙面全体（yref=paper）に1本引けば全段をまたぐ縦線になる
        fig.add_shape(
            type="line",
            x0=cursor,
            x1=cursor,
            xref="x",
            y0=0,
            y1=1,
            yref="paper",
            line={"color": _CURSOR_COLOR, "width": 1.5, "dash": "dash"},
        )
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    # マウスを乗せた時刻の値を全段まとめて表示し、段をまたぐ補助線を出す
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1)
    fig.update_layout(
        height=ROW_HEIGHT * n + 80,
        showlegend=False,
        hovermode="x unified",
        hoversubplots="axis",
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
        clickmode="event+select",
    )
    return fig


def _apply_y_axis(fig: go.Figure, row: int, view: SeriesView) -> None:
    """段の縦軸に範囲・対数軸を設定する。"""
    if view.log_y:
        axis: dict[str, Any] = {"type": "log"}
        if view.y_range is not None and min(view.y_range) > 0:
            # 対数軸の範囲は log10 の値で指定する
            axis["range"] = [math.log10(view.y_range[0]), math.log10(view.y_range[1])]
        fig.update_yaxes(row=row, col=1, **axis)
    elif view.y_range is not None:
        fig.update_yaxes(row=row, col=1, range=list(view.y_range))
