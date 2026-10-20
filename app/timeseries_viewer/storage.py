"""表形式ファイルを開く（CSV・Excel は Parquet にキャッシュする）。

数百MBのCSVを開くたびに読み直すと遅く、メモリも大きく使うため、初回に Parquet に変換して
キャッシュディレクトリに保存し、2回目以降はそれを開く。Parquet なら表示に必要な列・期間だけを
読める（列の選択と、pyarrow の述語プッシュダウン）。

- キャッシュは元ファイルの更新日時とサイズを記録した JSON と組で保存し、元ファイルが
  変わっていれば作り直す。
- 元ファイルは読むだけで、書き込まない（`data/raw` などの不変データも安全に開ける）。
- Parquet のファイルはキャッシュせずにそのまま開く。
"""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from util.csv_io import iter_csv_chunks
from util.paths import data_dir, sanitize_filename_component

SUPPORTED_SUFFIXES = (".csv", ".parquet", ".xlsx")

# 行の絞り込み条件: (列名, 演算子, 値)。演算子は "==", ">=", "<=" のいずれか
Filter = tuple[str, str, Any]

_OPERATORS: dict[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq,
    ">=": operator.ge,
    "<=": operator.le,
}


def _apply_filters(frame: pd.DataFrame, filters: Sequence[Filter] | None) -> pd.DataFrame:
    """絞り込み条件をすべて満たす行を返す（欠損との比較は満たさない扱い）。"""
    if not filters:
        return frame
    mask = pd.Series(True, index=frame.index)
    for column, op, value in filters:
        mask &= _OPERATORS[op](frame[column], value).fillna(False).astype(bool)
    return frame[mask]


@dataclass(frozen=True, eq=False)
class TableSource:
    """表のデータの読み出し口（Parquet ファイル、またはメモリ上の DataFrame）。

    列の型は pyarrow のスキーマで持ち、データは `read` で必要な列・行だけを読む。

    Attributes:
        schema: 列名と型（pyarrow のスキーマ）。
        path: Parquet ファイルのパス（メモリ上のデータならNone）。
        frame: メモリ上のデータ（Parquet ファイルならNone）。
    """

    schema: pa.Schema
    path: Path | None = None
    frame: pd.DataFrame | None = None

    @classmethod
    def from_parquet(cls, path: Path) -> TableSource:
        """Parquet ファイルを開く（データはまだ読まない）。"""
        return cls(pq.read_schema(path), path=path)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> TableSource:
        """メモリ上の DataFrame を、Parquet ファイルと同じ使い方で扱えるようにする。"""
        return cls(pa.Schema.from_pandas(frame, preserve_index=False), frame=frame)

    @property
    def columns(self) -> list[str]:
        """列名（並び順）。"""
        return list(self.schema.names)

    def column_type(self, column: str) -> pa.DataType:
        """列の型。"""
        return self.schema.field(column).type

    def read(self, columns: Sequence[str], filters: Sequence[Filter] | None = None) -> pd.DataFrame:
        """指定した列を、絞り込み条件を満たす行だけ読む。

        Parquet ファイルでは、条件を pyarrow に渡して不要な行を読まずに済ませる。
        条件の値と列の型が合わず pyarrow で比較できない場合は、読んでから pandas で絞り込む。

        Args:
            columns: 読む列（重複は1つにまとめる）。
            filters: 絞り込み条件（すべてを満たす行を残す）。

        Returns:
            読んだ DataFrame（インデックスは0始まりの連番）。
        """
        cols = list(dict.fromkeys(columns))
        if self.frame is not None:
            return _apply_filters(self.frame[cols], filters).reset_index(drop=True)
        assert self.path is not None
        if filters:
            try:
                return pd.read_parquet(self.path, columns=cols, filters=list(filters))
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError):
                # 例: 日付型（date32）の列を日時で比べる場合。全行を読んでから絞り込む
                pass
        frame = pd.read_parquet(self.path, columns=cols)
        return _apply_filters(frame, filters).reset_index(drop=True)


def default_cache_dir() -> Path:
    """キャッシュの既定の保存先（`data/interim/timeseries_viewer_cache`）。"""
    return data_dir() / "interim" / "timeseries_viewer_cache"


def cache_paths(source: Path, cache_dir: Path) -> tuple[Path, Path]:
    """元ファイルに対応するキャッシュの Parquet と、記録用 JSON のパス。

    ファイル名は元ファイル名に、絶対パスのハッシュを付けたもの（同名ファイルの衝突を避ける）。
    """
    digest = hashlib.sha1(str(source.resolve()).encode("utf-8")).hexdigest()[:10]
    stem = sanitize_filename_component(source.stem)
    return cache_dir / f"{stem}__{digest}.parquet", cache_dir / f"{stem}__{digest}.json"


def _signature(source: Path) -> dict[str, object]:
    stat = source.stat()
    return {"source": str(source.resolve()), "mtime_ns": stat.st_mtime_ns, "size": stat.st_size}


def is_cache_valid(source: Path, cache_dir: Path) -> bool:
    """キャッシュがあり、元ファイルが変わっていないか。"""
    parquet, meta = cache_paths(source, cache_dir)
    if not (parquet.is_file() and meta.is_file()):
        return False
    try:
        recorded = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(recorded == _signature(source))


def _iter_chunks(source: Path) -> Iterator[pd.DataFrame]:
    """元ファイルを DataFrame の塊として順に読む（変換用）。"""
    suffix = source.suffix.lower()
    if suffix == ".csv":
        # 大容量のCSVも全行をメモリに載せないよう、塊ごとに読む
        yield from iter_csv_chunks(source)
    elif suffix == ".xlsx":
        yield pd.read_excel(source)
    else:
        raise ValueError(f"対応していないファイル形式です: {source}")


def build_cache(source: Path, cache_dir: Path) -> Path:
    """元ファイルを Parquet に変換してキャッシュし、Parquet のパスを返す。

    塊ごとに pyarrow の ParquetWriter で追記する。書き込み途中のファイルが残らないよう、
    一時ファイルに書いてから置き換える。
    """
    parquet, meta = cache_paths(source, cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = parquet.with_suffix(".parquet.tmp")
    writer: pq.ParquetWriter | None = None
    try:
        for chunk in _iter_chunks(source):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema)
            else:
                # 塊ごとの型の揺れ（全て欠損の列など）は最初の塊の型にそろえる
                table = table.cast(writer.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        pq.write_table(pa.table({}), tmp)
    tmp.replace(parquet)
    meta.write_text(json.dumps(_signature(source), ensure_ascii=False), encoding="utf-8")
    return parquet


def open_table(source: Path, cache_dir: Path | None = None) -> TableSource:
    """ファイルを開く（CSV・Excel は必要ならキャッシュを作る）。

    Args:
        source: 開くファイル（CSV・Parquet・Excel）。
        cache_dir: キャッシュの保存先（Noneなら `default_cache_dir()`）。

    Returns:
        ファイルの内容を読み出す `TableSource`（データは必要になったときに読む）。

    Raises:
        ValueError: 対応していない形式の場合。
    """
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"対応していないファイル形式です: {source}")
    if suffix == ".parquet":
        return TableSource.from_parquet(source)
    cache_dir = cache_dir or default_cache_dir()
    if is_cache_valid(source, cache_dir):
        parquet = cache_paths(source, cache_dir)[0]
    else:
        parquet = build_cache(source, cache_dir)
    return TableSource.from_parquet(parquet)
