"""表形式ファイルを LazyFrame として開く（CSV・Excel は Parquet にキャッシュする）。

数百MBのCSVを開くたびに読み直すと遅く、メモリも大きく使うため、初回に Parquet に変換して
キャッシュディレクトリに保存し、2回目以降はそれを `pl.scan_parquet` で開く。Parquet なら
表示に必要な列・期間だけを読める（述語プッシュダウン）。

- キャッシュは元ファイルの更新日時とサイズを記録した JSON と組で保存し、元ファイルが
  変わっていれば作り直す。
- 元ファイルは読むだけで、書き込まない（`data/raw` などの不変データも安全に開ける）。
- Parquet のファイルはキャッシュせずにそのまま開く。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl

from util.csv_io import scan_csv_auto
from util.paths import data_dir, sanitize_filename_component

SUPPORTED_SUFFIXES = (".csv", ".parquet", ".xlsx")


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


def _to_lazy(source: Path) -> pl.LazyFrame:
    """元ファイルを LazyFrame として読む（変換用）。"""
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return scan_csv_auto(source)
    if suffix == ".xlsx":
        return pl.read_excel(source).lazy()
    raise ValueError(f"対応していないファイル形式です: {source}")


def build_cache(source: Path, cache_dir: Path) -> Path:
    """元ファイルを Parquet に変換してキャッシュし、Parquet のパスを返す。

    書き込み途中のファイルが残らないよう、一時ファイルに書いてから置き換える。
    """
    parquet, meta = cache_paths(source, cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = parquet.with_suffix(".parquet.tmp")
    # 大容量のCSVも全行をメモリに載せずに書き出す
    _to_lazy(source).sink_parquet(tmp)
    tmp.replace(parquet)
    meta.write_text(json.dumps(_signature(source), ensure_ascii=False), encoding="utf-8")
    return parquet


def open_table(source: Path, cache_dir: Path | None = None) -> pl.LazyFrame:
    """ファイルを LazyFrame として開く（CSV・Excel は必要ならキャッシュを作る）。

    Args:
        source: 開くファイル（CSV・Parquet・Excel）。
        cache_dir: キャッシュの保存先（Noneなら `default_cache_dir()`）。

    Returns:
        ファイルの内容の LazyFrame。

    Raises:
        ValueError: 対応していない形式の場合。
    """
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"対応していないファイル形式です: {source}")
    if suffix == ".parquet":
        return pl.scan_parquet(source)
    cache_dir = cache_dir or default_cache_dir()
    if is_cache_valid(source, cache_dir):
        parquet = cache_paths(source, cache_dir)[0]
    else:
        parquet = build_cache(source, cache_dir)
    return pl.scan_parquet(parquet)
