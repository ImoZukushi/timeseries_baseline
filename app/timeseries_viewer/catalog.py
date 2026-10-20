"""表示できるファイルの一覧と、列の種類（時刻・数値・グループ）の判定。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from timeseries_viewer.storage import SUPPORTED_SUFFIXES, TableSource

# グループ列（系列ID・地点名など）とみなす文字列列の最大の種類数
MAX_GROUP_VALUES = 500


@dataclass(frozen=True)
class DataFile:
    """表示の候補のファイル。

    Attributes:
        path: ファイルの絶対パス。
        label: 画面に出す名前（ルートの親からの相対パス。例: `data/raw/weather.csv`）。
        size: ファイルサイズ（バイト）。
    """

    path: Path
    label: str
    size: int


def discover_files(
    roots: Sequence[Path],
    exclude: Sequence[Path] = (),
    suffixes: Sequence[str] = SUPPORTED_SUFFIXES,
) -> list[DataFile]:
    """ルート以下の表形式ファイルを列挙する（ラベル順）。

    Args:
        roots: 探すディレクトリ（存在しないものは無視）。
        exclude: 除外するディレクトリ（キャッシュの保存先など）。
        suffixes: 対象の拡張子。

    Returns:
        見つかったファイル。ラベルは各ルートの親ディレクトリからの相対パス。
    """
    excluded = [p.resolve() for p in exclude]
    files: dict[Path, DataFile] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            resolved = path.resolve()
            if any(resolved.is_relative_to(e) for e in excluded):
                continue
            label = path.relative_to(root.parent).as_posix()
            files[resolved] = DataFile(resolved, label, path.stat().st_size)
    return sorted(files.values(), key=lambda f: f.label)


@dataclass(frozen=True)
class ColumnInfo:
    """ファイルの列の種類。

    Attributes:
        time_columns: 時刻列の候補（日時型。無ければ数値列）。
        value_columns: 値の列の候補（数値列）。
        group_columns: グループ列の候補（種類数の少ない文字列列）。
        default_group: 既定のグループ列（時刻だけでは行が重複し、時刻と組み合わせると
            一意になる列。パネル形式のデータの地点名など）。
    """

    time_columns: list[str]
    value_columns: list[str]
    group_columns: list[str]
    default_group: str | None


def _is_string(dtype: pa.DataType) -> bool:
    """文字列の列か（カテゴリ（辞書型）の文字列も含む）。"""
    if pa.types.is_dictionary(dtype):
        dtype = dtype.value_type
    return bool(pa.types.is_string(dtype) or pa.types.is_large_string(dtype))


def _is_numeric(dtype: pa.DataType) -> bool:
    """数値の列か（真偽値は含めない）。"""
    return bool(
        pa.types.is_integer(dtype) or pa.types.is_floating(dtype) or pa.types.is_decimal(dtype)
    )


def inspect_columns(table: TableSource) -> ColumnInfo:
    """ファイルの列の種類を調べる。

    既定のグループ列の判定は `eda.time_series_eda.find_grouping_column` と同じ考え方
    （時刻列に重複があり、時刻と組み合わせると一意になる列）で行う。読むのは時刻列と
    文字列の列だけにする。
    """
    types = {field.name: field.type for field in table.schema}
    datetimes = [c for c, t in types.items() if pa.types.is_timestamp(t) or pa.types.is_date(t)]
    numeric = [c for c, t in types.items() if _is_numeric(t)]
    strings = [c for c, t in types.items() if _is_string(t)]
    time_columns = datetimes or numeric
    group_columns: list[str] = []
    if strings:
        # 欠損も1つの値として数える
        counts = table.read(strings).nunique(dropna=False)
        group_columns = [c for c in strings if counts[c] <= MAX_GROUP_VALUES]
    default_group = None
    if datetimes and group_columns:
        time_col = datetimes[0]
        frame = table.read([time_col, *group_columns])
        if frame[time_col].nunique(dropna=False) < len(frame):
            default_group = next(
                (c for c in group_columns if not frame[[time_col, c]].duplicated().any()), None
            )
    return ColumnInfo(time_columns, numeric, group_columns, default_group)


def group_values(table: TableSource, group_col: str) -> list[str]:
    """グループ列の値の一覧（欠損を除き、文字列にして昇順）。"""
    values = table.read([group_col])[group_col].dropna().unique()
    return sorted(str(v) for v in values)
