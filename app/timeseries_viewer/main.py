"""時系列ビューアの画面（Streamlit）。

起動:
    uv run streamlit run app/timeseries_viewer/main.py

- サイドバーで系列（ファイル・時刻列・値の列・グループ）を最大5つ選ぶ
- 横軸（表示期間・描画点数）と縦軸（範囲・対数軸）をスライダなどで変える
- グラフの点をクリックすると、その時刻に全系列をまたぐ縦線（カーソル）を引き、
  各系列の最も近い観測の値を表に出す

環境変数 `TIMESERIES_VIEWER_ROOTS`（`os.pathsep` 区切り）で探すディレクトリを、
`TIMESERIES_VIEWER_CACHE_DIR` で Parquet キャッシュの保存先を変えられる（テスト用）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` ではこのファイルのディレクトリしか import パスに入らないため、src と app を加える
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "app"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import datetime as dt  # noqa: E402
import os  # noqa: E402
from typing import Any  # noqa: E402

import polars as pl  # noqa: E402
import streamlit as st  # noqa: E402

from timeseries_viewer.catalog import (  # noqa: E402
    ColumnInfo,
    DataFile,
    discover_files,
    group_values,
    inspect_columns,
)
from timeseries_viewer.figure import SeriesView, build_figure  # noqa: E402
from timeseries_viewer.series import (  # noqa: E402
    SeriesSpec,
    load_series,
    parse_axis_value,
    time_bounds,
    value_at,
    value_range,
)
from timeseries_viewer.storage import default_cache_dir, open_table  # noqa: E402
from util.paths import data_dir, outputs_dir  # noqa: E402

MAX_SERIES = 5
ROOTS_ENV = "TIMESERIES_VIEWER_ROOTS"
CACHE_ENV = "TIMESERIES_VIEWER_CACHE_DIR"
_NOT_SELECTED = "（選択してください）"
_NO_GROUP = "（なし）"


def _roots() -> tuple[str, ...]:
    env = os.environ.get(ROOTS_ENV)
    if env:
        return tuple(p for p in env.split(os.pathsep) if p)
    return (str(data_dir()), str(outputs_dir()))


def _cache_dir() -> str:
    return os.environ.get(CACHE_ENV) or str(default_cache_dir())


# --- データ読み込み（Streamlit のキャッシュ）--------------------------------------------------
# ファイルの更新日時（mtime_ns）を引数に含め、ファイルが変わったらキャッシュを使わないようにする


@st.cache_data(show_spinner=False)
def _files(roots: tuple[str, ...], cache_dir: str) -> list[DataFile]:
    return discover_files([Path(r) for r in roots], exclude=[Path(cache_dir)])


@st.cache_resource(show_spinner=False)
def _table(path: str, mtime_ns: int, cache_dir: str) -> pl.LazyFrame:
    return open_table(Path(path), Path(cache_dir))


def _lf(file: DataFile) -> pl.LazyFrame:
    return _table(str(file.path), file.path.stat().st_mtime_ns, _cache_dir())


@st.cache_data(show_spinner=False)
def _columns(path: str, mtime_ns: int, cache_dir: str) -> ColumnInfo:
    return inspect_columns(_table(path, mtime_ns, cache_dir))


@st.cache_data(show_spinner=False)
def _groups(path: str, mtime_ns: int, cache_dir: str, group_col: str) -> list[str]:
    return group_values(_table(path, mtime_ns, cache_dir), group_col)


def _key(file: DataFile) -> tuple[str, int, str]:
    return str(file.path), file.path.stat().st_mtime_ns, _cache_dir()


@st.cache_data(show_spinner=False)
def _bounds(key: tuple[str, int, str], spec: SeriesSpec) -> tuple[Any, Any] | None:
    return time_bounds(_table(*key), spec)


@st.cache_data(show_spinner=False)
def _values(
    key: tuple[str, int, str], spec: SeriesSpec, x_range: tuple[Any, Any], max_points: int
) -> pl.DataFrame:
    return load_series(_table(*key), spec, x_range, max_points)


@st.cache_data(show_spinner=False)
def _value_range(
    key: tuple[str, int, str], spec: SeriesSpec, x_range: tuple[Any, Any]
) -> tuple[float, float] | None:
    return value_range(_table(*key), spec, x_range)


@st.cache_data(show_spinner=False)
def _value_at(key: tuple[str, int, str], spec: SeriesSpec, t: Any) -> tuple[Any, float] | None:
    return value_at(_table(*key), spec, t)


# --- 画面の部品 ------------------------------------------------------------------------------


def _select_series(index: int, files: list[DataFile]) -> tuple[DataFile, SeriesSpec] | None:
    """サイドバーで1系列を選ぶ（選べていなければNone）。"""
    by_label = {f.label: f for f in files}
    with st.sidebar.expander(f"系列 {index + 1}", expanded=True):
        label = st.selectbox("ファイル", [_NOT_SELECTED, *by_label], key=f"file_{index}")
        if label == _NOT_SELECTED:
            return None
        file = by_label[label]
        with st.spinner(
            f"{label} を読み込み中"
            "（大きなファイルの初回は Parquet キャッシュを作るため時間がかかります）"
        ):
            info = _columns(*_key(file))
        if not info.time_columns or not info.value_columns:
            st.warning("時刻列または数値の列がありません")
            return None
        # 選択肢がファイルごとに変わるため、キーにファイル名を含める
        time_col = st.selectbox("時刻列", info.time_columns, key=f"time_{index}_{label}")
        value_options = [c for c in info.value_columns if c != time_col]
        if not value_options:
            st.warning("時刻列以外の数値の列がありません")
            return None
        value_col = st.selectbox("値の列", value_options, key=f"value_{index}_{label}")
        group_col, group_value = None, None
        if info.group_columns:
            options = [_NO_GROUP, *info.group_columns]
            default = options.index(info.default_group) if info.default_group in options else 0
            chosen = st.selectbox(
                "グループ列（地点・系列IDなど）",
                options,
                index=default,
                key=f"group_{index}_{label}",
                help="1つのファイルに複数の地点・系列が混在する場合に、表示する1つを選びます",
            )
            if chosen != _NO_GROUP:
                values = _groups(*_key(file), chosen)
                group_col = chosen
                group_value = st.selectbox(
                    f"{chosen} の値", values, key=f"group_value_{index}_{label}_{chosen}"
                )
    return file, SeriesSpec(label, time_col, value_col, group_col, group_value)


def _x_controls(bounds: list[tuple[Any, Any]]) -> tuple[tuple[Any, Any], int]:
    """横軸（表示期間・描画点数）の操作部品。"""
    lo, hi = min(b[0] for b in bounds), max(b[1] for b in bounds)
    col_range, col_points = st.columns([4, 1])
    with col_range:
        if isinstance(lo, dt.datetime):
            # 刻み幅は全期間の1/1000（最小1秒）
            step = max((hi - lo) / 1000, dt.timedelta(seconds=1))
            x_range = st.slider(
                "表示期間（横軸）",
                min_value=lo,
                max_value=hi,
                value=(lo, hi),
                step=step,
                format="YYYY-MM-DD HH:mm:ss",
                key=f"x_range_{lo}_{hi}",
            )
        else:
            lo, hi = float(lo), float(hi)
            x_range = st.slider(
                "表示範囲（横軸）",
                min_value=lo,
                max_value=hi,
                value=(lo, hi),
                step=(hi - lo) / 1000 or 1.0,
                key=f"x_range_{lo}_{hi}",
            )
    with col_points:
        max_points = st.slider(
            "最大描画点数（1系列）",
            500,
            20000,
            5000,
            step=500,
            help="表示期間の点がこれより多い場合、区間ごとの最小・最大の点に間引いて描きます",
        )
    return (x_range[0], x_range[1]), max_points


def _y_controls(
    index: int, spec: SeriesSpec, vrange: tuple[float, float] | None
) -> tuple[tuple[float, float] | None, bool]:
    """1系列分の縦軸（範囲・対数軸）の操作部品。"""
    col_label, col_mode, col_range, col_log = st.columns([3, 2, 6, 1.5])
    col_label.markdown(f"**{index + 1}. {spec.label}**")
    if vrange is None:
        col_range.caption("表示期間に値がありません")
        return None, False
    mode = col_mode.radio(
        "縦軸",
        ["自動", "手動"],
        horizontal=True,
        key=f"y_mode_{index}_{spec}",
        label_visibility="collapsed",
    )
    log_y = col_log.checkbox(
        "対数",
        key=f"y_log_{index}_{spec}",
        disabled=vrange[0] <= 0,
        help="値がすべて正の場合のみ使えます" if vrange[0] <= 0 else None,
    )
    if mode == "自動":
        return None, log_y
    lo, hi = vrange
    span = (hi - lo) or abs(hi) or 1.0
    y_range = col_range.slider(
        "縦軸の範囲",
        min_value=float(lo - span),
        max_value=float(hi + span),
        value=(float(lo), float(hi)),
        step=span / 200,
        key=f"y_range_{index}_{spec}_{lo}_{hi}",
        label_visibility="collapsed",
    )
    return (y_range[0], y_range[1]), log_y


def _update_cursor_from_click(chart_key: str, is_datetime: bool) -> None:
    """グラフのクリック（点の選択）があれば、その時刻をカーソルにする。"""
    state = st.session_state.get(chart_key)
    points = (state or {}).get("selection", {}).get("points", []) if state else []
    if not points:
        return
    raw = points[0].get("x")
    # 同じ選択のまま再描画された場合は、手入力したカーソルを上書きしない
    if raw is None or raw == st.session_state.get("last_click"):
        return
    st.session_state["last_click"] = raw
    try:
        st.session_state["cursor"] = parse_axis_value(raw, is_datetime)
    except ValueError:
        st.warning(f"クリックした位置の時刻を読み取れませんでした: {raw}")


def _cursor_controls(is_datetime: bool) -> None:
    """カーソルの手入力・消去の部品。"""
    col_input, col_set, col_clear = st.columns([4, 1, 1])
    current = st.session_state.get("cursor")
    text = col_input.text_input(
        "カーソルの時刻（グラフの点をクリックしても設定できます）",
        value="" if current is None else str(current),
        placeholder="2016-01-01 12:00:00" if is_datetime else "数値",
    )
    if col_set.button("設定", width="stretch") and text:
        try:
            st.session_state["cursor"] = parse_axis_value(text.strip(), is_datetime)
        except ValueError:
            st.error(f"時刻として読み取れません: {text}")
    if col_clear.button("カーソルを消す", width="stretch"):
        st.session_state["cursor"] = None
        st.session_state["last_click"] = None
        # グラフの選択状態も消すため、グラフのキーを変える
        st.session_state["chart_nonce"] = st.session_state.get("chart_nonce", 0) + 1


def _format_delta(delta: Any) -> str:
    if isinstance(delta, dt.timedelta):
        return str(delta)
    return f"{delta:.6g}"


# --- 画面 ------------------------------------------------------------------------------------


def main() -> None:
    """時系列ビューアの画面を描く。"""
    st.set_page_config(page_title="時系列ビューア", layout="wide")
    st.title("時系列ビューア")
    files = _files(_roots(), _cache_dir())
    st.sidebar.header("系列の選択")
    st.sidebar.caption(
        f"対象: {', '.join(Path(r).name + '/' for r in _roots())}（{len(files)}ファイル）"
    )
    if not files:
        st.info("表示できるファイル（CSV・Parquet・Excel）が見つかりません")
        return
    n_series = st.sidebar.slider("表示する系列数", 1, MAX_SERIES, 1)
    selected = [s for i in range(n_series) if (s := _select_series(i, files)) is not None]
    if not selected:
        st.info("サイドバーでファイルと列を選んでください")
        return

    bounds = {}
    for file, spec in selected:
        b = _bounds(_key(file), spec)
        if b is None:
            st.warning(f"{spec.label}: データがありません")
        else:
            bounds[spec] = b
    # 横軸の種類（日時 / 数値）は最初の系列に合わせ、違う種類の系列は表示しない
    kinds = {spec: isinstance(b[0], dt.datetime) for spec, b in bounds.items()}
    if not kinds:
        return
    is_datetime = next(iter(kinds.values()))
    for spec, kind in kinds.items():
        if kind != is_datetime:
            st.warning(
                f"{spec.label}: 横軸の種類（日時 / 数値）が1つ目の系列と違うため表示しません"
            )
    shown = [(f, s) for f, s in selected if s in kinds and kinds[s] == is_datetime]

    x_range, max_points = _x_controls([bounds[s] for _, s in shown])
    with st.expander("縦軸の設定（系列ごと）", expanded=False):
        y_settings = [
            _y_controls(i, spec, _value_range(_key(file), spec, x_range))
            for i, (file, spec) in enumerate(shown)
        ]

    chart_key = f"chart_{st.session_state.get('chart_nonce', 0)}"
    _update_cursor_from_click(chart_key, is_datetime)
    _cursor_controls(is_datetime)
    cursor = st.session_state.get("cursor")
    if cursor is not None and isinstance(cursor, dt.datetime) != is_datetime:
        cursor = None  # 横軸の種類が変わった場合は使わない

    views = []
    nearest_rows = []
    for (file, spec), (y_range, log_y) in zip(shown, y_settings, strict=True):
        point = _value_at(_key(file), spec, cursor) if cursor is not None else None
        data = _values(_key(file), spec, x_range, max_points)
        views.append(SeriesView(spec.label, data, y_range, log_y, point))
        if cursor is not None:
            nearest_rows.append(
                {
                    "系列": spec.label,
                    "最も近い観測の時刻": "－" if point is None else str(point[0]),
                    "値": None if point is None else point[1],
                    "カーソルとの差": "－" if point is None else _format_delta(point[0] - cursor),
                }
            )
    fig = build_figure(views, x_range=x_range, cursor=cursor)
    st.plotly_chart(
        fig,
        key=chart_key,
        on_select="rerun",
        selection_mode="points",
        config={"displaylogo": False},
    )
    if cursor is not None:
        st.subheader(f"カーソル {cursor} での値")
        st.dataframe(pl.DataFrame(nearest_rows), hide_index=True, width="stretch")
    st.caption(
        "点をクリックするとカーソル（赤の破線）を移動します。マウスを乗せると、その時刻の全系列の値が"
        "表示されます。点数が多い期間は区間ごとの最小・最大に間引いて描いています（カーソルの値は"
        "間引く前のデータから求めます）。"
    )


main()
