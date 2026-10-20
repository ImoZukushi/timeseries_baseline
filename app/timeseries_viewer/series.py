"""系列の指定と、表示範囲の切り出し・間引き・指定時刻の値。

1つの系列は「ファイル・時刻列・値の列（・グループ列とその値）」で決まる。データは
`storage.TableSource` から、必要な列・期間・グループの行だけを読む。

時刻列が日付型の場合は日時型（マイクロ秒）にそろえる。時刻列が数値の場合は数値の横軸として扱う。
戻り値の時刻は Python の `datetime`（数値の横軸なら `int` / `float`）にする。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from timeseries_viewer.storage import Filter, TableSource

TIME, VALUE = "time", "value"

# 横軸の値（日時または数値）
AxisValue = dt.datetime | float


@dataclass(frozen=True)
class SeriesSpec:
    """1つの系列の指定。

    Attributes:
        file: ファイルのラベル（`catalog.DataFile.label`）。
        time_col: 時刻列。
        value_col: 値の列。
        group_col: グループ列（パネル形式のデータで地点などを選ぶ場合）。
        group_value: グループ列で選ぶ値（文字列として比較する）。
    """

    file: str
    time_col: str
    value_col: str
    group_col: str | None = None
    group_value: str | None = None

    @property
    def label(self) -> str:
        """画面・凡例に出す名前（ファイル名 / 列 / グループ）。"""
        name = self.file.rsplit("/", 1)[-1]
        parts = [name, self.value_col]
        if self.group_col is not None and self.group_value is not None:
            parts.append(f"{self.group_col}={self.group_value}")
        return " / ".join(parts)


def _to_python(value: Any) -> Any:
    """pandas・numpy のスカラーを Python の値にする（日時は `datetime`）。"""
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_string(dtype: pa.DataType) -> bool:
    if pa.types.is_dictionary(dtype):
        dtype = dtype.value_type
    return bool(pa.types.is_string(dtype) or pa.types.is_large_string(dtype))


def _pushdown_filters(
    table: TableSource, spec: SeriesSpec, x_range: tuple[Any, Any] | None
) -> list[Filter]:
    """読む前に pyarrow に渡せる絞り込み条件（グループの値・表示範囲）。"""
    filters: list[Filter] = []
    if (
        spec.group_col is not None
        and spec.group_value is not None
        and _is_string(table.column_type(spec.group_col))
    ):
        filters.append((spec.group_col, "==", spec.group_value))
    time_type = table.column_type(spec.time_col)
    if x_range is not None and (
        pa.types.is_timestamp(time_type)
        or pa.types.is_integer(time_type)
        or pa.types.is_floating(time_type)
    ):
        filters += [(spec.time_col, ">=", x_range[0]), (spec.time_col, "<=", x_range[1])]
    return filters


def select_series(
    table: TableSource, spec: SeriesSpec, x_range: tuple[Any, Any] | None = None
) -> pd.DataFrame:
    """系列の時刻・値の2列（列名 `time`, `value`）を取り出す（時刻が欠損の行は除く）。

    Args:
        table: ファイルの `TableSource`。
        spec: 系列の指定。
        x_range: 表示範囲（両端を含む）。Noneなら全期間。

    Returns:
        列 `time`, `value` の DataFrame（行の並びはファイルのまま）。
    """
    columns = [spec.time_col, spec.value_col]
    grouped = spec.group_col is not None and spec.group_value is not None
    if grouped:
        assert spec.group_col is not None
        columns.append(spec.group_col)
    frame = table.read(columns, _pushdown_filters(table, spec, x_range))
    if grouped:
        # グループの値は文字列として比べる（欠損の行は選ばない）
        group = frame[spec.group_col]
        frame = frame[group.notna() & (group.astype(str) == spec.group_value)]
    time = frame[spec.time_col]
    if not (pd.api.types.is_numeric_dtype(time) or pd.api.types.is_datetime64_any_dtype(time)):
        # 日付型（Parquet の date32 は Python の date の列になる）は日時（マイクロ秒）にそろえる
        time = pd.to_datetime(time).astype("datetime64[us]")
    data = pd.DataFrame(
        {TIME: time.to_numpy(), VALUE: frame[spec.value_col].astype("float64").to_numpy()}
    )
    data = data[data[TIME].notna()]
    if x_range is not None:
        data = data[(data[TIME] >= x_range[0]) & (data[TIME] <= x_range[1])]
    return data.reset_index(drop=True)


def time_bounds(table: TableSource, spec: SeriesSpec) -> tuple[Any, Any] | None:
    """系列の時刻の最小・最大（データが無ければNone）。"""
    times = select_series(table, spec)[TIME]
    if times.empty:
        return None
    return _to_python(times.min()), _to_python(times.max())


def value_range(
    table: TableSource, spec: SeriesSpec, x_range: tuple[Any, Any] | None = None
) -> tuple[float, float] | None:
    """表示範囲内の値の最小・最大（有限の値が無ければNone）。"""
    values = select_series(table, spec, x_range)[VALUE].to_numpy()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(values.min()), float(values.max())


def load_series(
    table: TableSource,
    spec: SeriesSpec,
    x_range: tuple[Any, Any] | None = None,
    max_points: int | None = 5000,
) -> pd.DataFrame:
    """表示範囲の系列を時刻順に取り出す（点が多ければ間引く）。

    間引きは、時刻順に並べた点を `max_points // 2` 個の区間に分け、各区間の最小値と最大値の
    点だけを残す。単純な等間隔の抽出と違い、急な山・谷（外れ値やピーク）が消えない。
    間引く場合、値が欠損の点は除く（線の途切れは図の側で時刻の空白から判断する）。

    Args:
        table: ファイルの `TableSource`。
        spec: 系列の指定。
        x_range: 表示範囲（両端を含む）。Noneなら全期間。
        max_points: 最大の点数（Noneなら間引かない）。

    Returns:
        列 `time`, `value` の DataFrame。
    """
    data = select_series(table, spec, x_range).sort_values(TIME, kind="stable")
    data = data.reset_index(drop=True)
    if max_points is None or len(data) <= max_points:
        return data
    n_buckets = max(1, max_points // 2)
    valid = data[np.isfinite(data[VALUE].to_numpy())].reset_index(drop=True)
    n_valid = len(valid)
    if n_valid == 0:
        return valid
    # 各区間の最小値・最大値の点（行番号。同じ値なら先の行）だけを残す
    bucket = np.arange(n_valid, dtype=np.int64) * n_buckets // n_valid
    by_bucket = valid[VALUE].groupby(bucket)
    keep = np.union1d(by_bucket.idxmin().to_numpy(), by_bucket.idxmax().to_numpy())
    return valid.loc[keep, [TIME, VALUE]].reset_index(drop=True)


def value_at(table: TableSource, spec: SeriesSpec, t: Any) -> tuple[Any, float] | None:
    """時刻 t に最も近い観測（値が欠損でない点）の (時刻, 値)。間引く前のデータから求める。

    前後の観測が同じだけ離れている場合は前の観測を返す。データが無ければNone。
    """
    data = select_series(table, spec)
    data = data[data[VALUE].notna()]
    before = data.loc[data[TIME] <= t, TIME].max()
    after = data.loc[data[TIME] >= t, TIME].min()
    candidates = [_to_python(x) for x in (before, after) if not pd.isna(x)]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda x: abs(x - t))
    value = data.loc[data[TIME] == nearest, VALUE].iloc[0]
    return nearest, float(value)


def parse_axis_value(raw: Any, is_datetime: bool) -> Any:
    """横軸の値（Plotly のクリックや手入力から文字列で届く）を日時または数値にする。

    Raises:
        ValueError: 日時・数値として読み取れない場合。
    """
    if not is_datetime:
        return float(raw)
    if isinstance(raw, dt.datetime):
        return raw
    return dt.datetime.fromisoformat(str(raw).strip())
