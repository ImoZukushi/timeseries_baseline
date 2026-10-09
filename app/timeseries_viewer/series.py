"""系列の指定と、表示範囲の切り出し・間引き・指定時刻の値。

1つの系列は「ファイル・時刻列・値の列（・グループ列とその値）」で決まる。データは
LazyFrame（`storage.open_table`）のまま扱い、必要な期間・列だけを読む。

時刻列が日付型の場合は日時型（マイクロ秒）にそろえる。時刻列が数値の場合は数値の横軸として扱う。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

import polars as pl

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


def select_series(lf: pl.LazyFrame, spec: SeriesSpec) -> pl.LazyFrame:
    """系列の時刻・値の2列（列名 `time`, `value`）を取り出す（時刻が欠損の行は除く）。"""
    if spec.group_col is not None and spec.group_value is not None:
        lf = lf.filter(pl.col(spec.group_col).cast(pl.String) == spec.group_value)
    time = pl.col(spec.time_col)
    if lf.collect_schema()[spec.time_col] == pl.Date:
        time = time.cast(pl.Datetime("us"))
    return lf.select(time.alias(TIME), pl.col(spec.value_col).cast(pl.Float64).alias(VALUE)).filter(
        pl.col(TIME).is_not_null()
    )


def _in_range(lf: pl.LazyFrame, x_range: tuple[Any, Any] | None) -> pl.LazyFrame:
    if x_range is None:
        return lf
    return lf.filter(pl.col(TIME).is_between(x_range[0], x_range[1]))


def time_bounds(lf: pl.LazyFrame, spec: SeriesSpec) -> tuple[Any, Any] | None:
    """系列の時刻の最小・最大（データが無ければNone）。"""
    row = (
        select_series(lf, spec)
        .select(pl.col(TIME).min().alias("min"), pl.col(TIME).max().alias("max"))
        .collect()
        .row(0)
    )
    return None if row[0] is None else (row[0], row[1])


def value_range(
    lf: pl.LazyFrame, spec: SeriesSpec, x_range: tuple[Any, Any] | None = None
) -> tuple[float, float] | None:
    """表示範囲内の値の最小・最大（有限の値が無ければNone）。"""
    values = _in_range(select_series(lf, spec), x_range).filter(pl.col(VALUE).is_finite())
    row = (
        values.select(pl.col(VALUE).min().alias("min"), pl.col(VALUE).max().alias("max"))
        .collect()
        .row(0)
    )
    return None if row[0] is None else (float(row[0]), float(row[1]))


def load_series(
    lf: pl.LazyFrame,
    spec: SeriesSpec,
    x_range: tuple[Any, Any] | None = None,
    max_points: int | None = 5000,
) -> pl.DataFrame:
    """表示範囲の系列を時刻順に取り出す（点が多ければ間引く）。

    間引きは、時刻順に並べた点を `max_points // 2` 個の区間に分け、各区間の最小値と最大値の
    点だけを残す。単純な等間隔の抽出と違い、急な山・谷（外れ値やピーク）が消えない。
    間引く場合、値が欠損の点は除く（線の途切れは図の側で時刻の空白から判断する）。

    Args:
        lf: ファイルの LazyFrame。
        spec: 系列の指定。
        x_range: 表示範囲（両端を含む）。Noneなら全期間。
        max_points: 最大の点数（Noneなら間引かない）。

    Returns:
        列 `time`, `value` の DataFrame。
    """
    data = _in_range(select_series(lf, spec), x_range).sort(TIME)
    n = data.select(pl.len()).collect().item()
    if max_points is None or n <= max_points:
        return data.collect()
    n_buckets = max(1, max_points // 2)
    valid = data.filter(pl.col(VALUE).is_finite()).with_row_index("__i")
    n_valid = valid.select(pl.len()).collect().item()
    if n_valid == 0:
        return pl.DataFrame(schema=data.collect_schema())
    # 行番号は u32 のため、掛け算のあふれを避けて Int64 で計算する
    bucketed = valid.with_columns(
        (pl.col("__i").cast(pl.Int64) * n_buckets // n_valid).alias("__b")
    )
    # 各区間の最小値・最大値の点（行番号）を求め、その行だけを残す
    extremes = bucketed.group_by("__b").agg(
        pl.col("__i").get(pl.col(VALUE).arg_min()).alias("__min"),
        pl.col("__i").get(pl.col(VALUE).arg_max()).alias("__max"),
    )
    keep = pl.concat(
        [
            extremes.select(pl.col("__min").alias("__i")),
            extremes.select(pl.col("__max").alias("__i")),
        ]
    ).unique()
    return valid.join(keep, on="__i", how="semi").sort("__i").select(TIME, VALUE).collect()


def value_at(lf: pl.LazyFrame, spec: SeriesSpec, t: Any) -> tuple[Any, float] | None:
    """時刻 t に最も近い観測（値が欠損でない点）の (時刻, 値)。間引く前のデータから求める。

    前後の観測が同じだけ離れている場合は前の観測を返す。データが無ければNone。
    """
    data = select_series(lf, spec).filter(pl.col(VALUE).is_not_null())
    before, after = (
        data.select(
            pl.col(TIME).filter(pl.col(TIME) <= t).max().alias("before"),
            pl.col(TIME).filter(pl.col(TIME) >= t).min().alias("after"),
        )
        .collect()
        .row(0)
    )
    candidates = [x for x in (before, after) if x is not None]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda x: abs(x - t))
    value = data.filter(pl.col(TIME) == nearest).select(pl.col(VALUE).first()).collect().item()
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
