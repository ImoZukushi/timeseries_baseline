"""エンコーディング自動判定つきでCSVを読み込む汎用ユーティリティ（pandas）。

前提・制約:
    対象とするCSVはUTF-8またはCP932（Shift-JIS）のいずれかでエンコードされている
    ことを前提とする。30MBを超える大容量ファイルのうち、ヘッダ行を除くデータ本体が
    完全にASCII文字のみで構成されている場合に限り、全文デコードを行わずに読み込む
    （本プロジェクトの `wind_*.csv` が該当）。大容量かつデータ本体に非ASCII
    文字を含むファイルは全文デコードするため、メモリ使用量が大きくなる点に注意すること。
    観測データの欠測記号として `*`（アスタリスク）が使われている列があることを実データで
    確認したため、これを欠損値として扱う（`wind_*.csv` の風速・風向列が該当）。
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

import pandas as pd

# 大容量ファイル判定の閾値。これを超える場合はヘッダ行のみをデコードし、
# データ本体がASCII文字のみで構成されていることを確認できたときに限り
# 全文デコードせずに読み込む。
LARGE_FILE_THRESHOLD_BYTES = 30 * 1024 * 1024  # 30MB

# 欠測を表す記号として扱う値。wind_*.csv の風速・風向列で "*" が欠測記号として
# 使われていることを実データで確認したため、欠損値として読み込む。
# 空文字も欠損とする（"NA" などの文字列は値として残す。pandas の既定の欠損記号は使わない）
NULL_VALUE_MARKERS = ["*"]
_NA_VALUES = ["", *NULL_VALUE_MARKERS]

# 日時列とみなす閾値。文字列列のうち、この割合以上がdatetime変換に成功すれば
# datetime型の列として扱う。
DATETIME_PARSE_SUCCESS_THRESHOLD = 0.95

_CANDIDATE_ENCODINGS = ("utf-8", "cp932")

# 日時変換を試みる際に明示的に試す書式一覧。書式の自動推定は、日時らしくない文字列
# （地名など）を誤って変換するおそれがあるため使わず、固定の書式候補に厳密一致のみ試す。
_DATETIME_FORMATS = (
    # ISO 8601（小数秒あり・なし）
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
)

# 大容量CSVを分割して読むときの1回あたりの行数
DEFAULT_CHUNKSIZE = 200_000


def detect_encoding(path: Path, sample_size: int = 1_000_000) -> str:
    """ファイル先頭のバイト列からエンコーディングを推定する。

    Args:
        path: 対象CSVファイルのパス。
        sample_size: 判定に使用する先頭バイト数。

    Returns:
        "utf-8" または "cp932"。いずれでもデコードできない場合は "cp932" を返す
        （呼び出し側で置換文字を許容して読み込む前提）。
    """
    with path.open("rb") as f:
        sample = f.read(sample_size)
    for encoding in _CANDIDATE_ENCODINGS:
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "cp932"


def _is_ascii_only_from(path: Path, skip_bytes: int, chunk_size: int = 8 * 1024 * 1024) -> bool:
    """指定バイト位置以降の内容がすべてASCII文字のみかどうかを判定する。

    大容量ファイルを全文デコードせずに安全に読めるかどうかの判断に使う。

    Args:
        path: 対象ファイルのパス。
        skip_bytes: 判定を開始するバイトオフセット（ヘッダ行の長さなど）。
        chunk_size: 一度に読み込むバイト数。

    Returns:
        指定位置以降がすべてASCII文字であれば True。
    """
    with path.open("rb") as f:
        f.seek(skip_bytes)
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                return True
            try:
                chunk.decode("ascii")
            except UnicodeDecodeError:
                return False


def _is_text(series: pd.Series) -> bool:
    """文字列の列か（pandas 3 の `str` 型と、文字列を入れた object 型）。"""
    return pd.api.types.is_string_dtype(series) or series.dtype == object


def _find_datetime_format(non_null: pd.Series) -> str | None:
    """明示的な書式候補を順に試し、十分な一致率が得られる書式を探す。

    Args:
        non_null: 欠損を除いた文字列Series。

    Returns:
        サンプルが完全一致し、かつ全体でも `DATETIME_PARSE_SUCCESS_THRESHOLD`
        以上一致する書式文字列。条件を満たす書式がなければ None。
    """
    values = non_null.astype(str)
    sample = values.head(min(100, len(values)))
    for fmt in _DATETIME_FORMATS:
        sample_parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
        if sample_parsed.isna().any():
            continue  # サンプル内に書式不一致があれば別の書式を試す
        parsed = pd.to_datetime(values, format=fmt, errors="coerce")
        success_rate = parsed.notna().sum() / len(values)
        if success_rate >= DATETIME_PARSE_SUCCESS_THRESHOLD:
            return fmt
    return None


def _to_datetime(series: pd.Series, fmt: str) -> pd.Series:
    """書式を指定して日時に変換する（一致しない値は欠損。精度はマイクロ秒にそろえる）。"""
    return pd.to_datetime(series, format=fmt, errors="coerce").astype("datetime64[us]")


def _parse_datetime_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """文字列列のうち日時とみなせるものを日時型に変換する。

    Args:
        df: 変換対象のDataFrame。

    Returns:
        日時らしき文字列列を日時型に変換したDataFrame。
    """
    for col in df.columns:
        if not _is_text(df[col]):
            continue
        non_null = df[col].dropna()
        if non_null.empty:
            continue
        fmt = _find_datetime_format(non_null)
        if fmt is None:
            continue
        df[col] = _to_datetime(df[col], fmt)
    return df


def _read(source: Any, **kwargs: Any) -> pd.DataFrame:
    """欠損の記号をそろえて pandas で CSV を読む。"""
    return pd.read_csv(source, na_values=_NA_VALUES, keep_default_na=False, **kwargs)


def _large_ascii_header(path: Path) -> list[str] | None:
    """大容量でヘッダ行以降がASCII文字だけのCSVなら、ヘッダの列名を返す（それ以外は None）。

    ヘッダはUTF-8またはCP932でデコードする。
    """
    if path.stat().st_size <= LARGE_FILE_THRESHOLD_BYTES:
        return None
    with path.open("rb") as f:
        header_line = f.readline()
    try:
        header_text = header_line.decode("utf-8")
    except UnicodeDecodeError:
        header_text = header_line.decode("cp932", errors="replace")
    if not _is_ascii_only_from(path, skip_bytes=len(header_line)):
        return None
    return header_text.lstrip("﻿").rstrip("\r\n").split(",")


ColumnKind = Literal["datetime", "numeric", "text"]


def _column_kinds(sample: pd.DataFrame) -> dict[str, tuple[ColumnKind, str | None]]:
    """先頭のサンプルから、列ごとの種類（日時と書式・数値・文字列）を決める。"""
    kinds: dict[str, tuple[ColumnKind, str | None]] = {}
    for col in sample.columns:
        series = sample[col]
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            kinds[col] = ("numeric", None)
            continue
        non_null = series.dropna()
        fmt = _find_datetime_format(non_null) if not non_null.empty else None
        kinds[col] = ("datetime", fmt) if fmt is not None else ("text", None)
    return kinds


def _apply_kinds(
    chunk: pd.DataFrame, kinds: dict[str, tuple[ColumnKind, str | None]]
) -> pd.DataFrame:
    """列の種類に合わせて型をそろえる（分割した各部分で型がぶれないようにする）。"""
    out: dict[str, pd.Series] = {}
    for col, (kind, fmt) in kinds.items():
        series = chunk[col]
        if kind == "numeric":
            out[col] = pd.to_numeric(series, errors="coerce").astype("float64")
        elif kind == "datetime":
            assert fmt is not None
            out[col] = _to_datetime(series.astype("string"), fmt)
        else:
            out[col] = series.astype("string")
    return pd.DataFrame(out, index=chunk.index)


def iter_csv_chunks(
    path: Path, chunksize: int = DEFAULT_CHUNKSIZE, datetime_sample_rows: int = 1000
) -> Iterator[pd.DataFrame]:
    """CSVファイルを、型をそろえた DataFrame の列（分割読み込み）として読む。

    大容量でデータ本体がASCII文字のみのCSV（本プロジェクトの `wind_*.csv` など）は、
    `chunksize` 行ずつ読み、全行をメモリに載せずに Parquet への書き出しなどができる。
    各列の種類（日時と書式・数値・文字列）は先頭 `datetime_sample_rows` 行で決め、すべての
    部分で同じ型（数値は float64、文字列は string、日時はマイクロ秒精度）にそろえる。
    それ以外のファイルは `read_csv_auto(path)` の結果を1つだけ返す。

    Args:
        path: 読み込むCSVファイルのパス。
        chunksize: 1回に読む行数（大容量ファイルのみ）。
        datetime_sample_rows: 列の種類の判定に使う先頭の行数（大容量ファイルのみ）。

    Yields:
        読み込んだ DataFrame（大容量ファイルでなければ1つ）。
    """
    header = _large_ascii_header(path)
    if header is None:
        yield read_csv_auto(path)
        return
    common: dict[str, Any] = {
        "header": None,
        "skiprows": 1,
        "names": header,
        "encoding": "utf-8",
        "encoding_errors": "replace",
    }
    kinds = _column_kinds(_read(path, nrows=datetime_sample_rows, **common))
    reader = pd.read_csv(
        path, na_values=_NA_VALUES, keep_default_na=False, chunksize=chunksize, **common
    )
    for chunk in reader:
        yield _apply_kinds(chunk, kinds)


def read_csv_auto(path: Path) -> pd.DataFrame:
    """CSVファイルをエンコーディング自動判定つきで読み込む。

    UTF-8で読めない場合はCP932（Shift-JIS）として扱う。日時らしき文字列列は
    自動的に日時型へ変換する。

    Args:
        path: 読み込むCSVファイルのパス。

    Returns:
        読み込んだDataFrame。
    """
    encoding = detect_encoding(path)
    file_size = path.stat().st_size

    if encoding == "utf-8":
        df = _read(path, encoding="utf-8-sig")
    elif file_size <= LARGE_FILE_THRESHOLD_BYTES:
        text = path.read_bytes().decode("cp932", errors="replace")
        df = _read(io.StringIO(text))
    else:
        header = _large_ascii_header(path)
        if header is not None:
            df = _read(
                path,
                header=None,
                skiprows=1,
                names=header,
                encoding="utf-8",
                encoding_errors="replace",
            )
        else:
            text = path.read_bytes().decode("cp932", errors="replace")
            df = _read(io.StringIO(text))

    return _parse_datetime_like_columns(df)
