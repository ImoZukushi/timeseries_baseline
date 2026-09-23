"""エンコーディング自動判定つきでCSVを読み込む汎用ユーティリティ。

前提・制約:
    対象とするCSVはUTF-8またはCP932（Shift-JIS）のいずれかでエンコードされている
    ことを前提とする。30MBを超える大容量ファイルのうち、ヘッダ行を除くデータ本体が
    完全にASCII文字のみで構成されている場合に限り、全文デコードを行わずlazy scanで
    読み込む（本プロジェクトの `wind_*.csv` が該当）。大容量かつデータ本体に非ASCII
    文字を含むファイルは全文デコードするため、メモリ使用量が大きくなる点に注意すること。
    観測データの欠測記号として `*`（アスタリスク）が使われている列があることを実データで
    確認したため、これをnull値として扱う（`wind_*.csv` の風速・風向列が該当）。
"""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl

# 大容量ファイル判定の閾値。これを超える場合はヘッダ行のみをデコードし、
# データ本体がASCII文字のみで構成されていることを確認できたときに限り
# lazy scanでメモリ効率よく読み込む。
LARGE_FILE_THRESHOLD_BYTES = 30 * 1024 * 1024  # 30MB

# 欠測を表す記号として扱う値。wind_*.csv の風速・風向列で "*" が欠測記号として
# 使われていることを実データで確認したため、null値として読み込む。
NULL_VALUE_MARKERS = ["*"]

# 日時列とみなす閾値。文字列列のうち、この割合以上がdatetime変換に成功すれば
# datetime型の列として扱う。
DATETIME_PARSE_SUCCESS_THRESHOLD = 0.95

_CANDIDATE_ENCODINGS = ("utf-8", "cp932")


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

    大容量ファイルを全文デコードせずにlazy scanで安全に読めるかどうかの判断に使う。

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


# 日時変換を試みる際に明示的に試す書式一覧。
# polarsの書式自動推定（to_datetime(strict=False) を書式なしで呼ぶ方法）は、
# 日時らしくない文字列（地名など）に対して稀に内部でクラッシュする挙動が
# 確認されたため使用せず、固定の書式候補に対して厳密一致のみ試す。
_DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
)


def _find_datetime_format(non_null: pl.Series) -> str | None:
    """明示的な書式候補を順に試し、十分な一致率が得られる書式を探す。

    Args:
        non_null: 欠損を除いた文字列Series。

    Returns:
        サンプルが完全一致し、かつ全体でも `DATETIME_PARSE_SUCCESS_THRESHOLD`
        以上一致する書式文字列。条件を満たす書式がなければ None。
    """
    sample = non_null.head(min(100, non_null.len()))
    for fmt in _DATETIME_FORMATS:
        try:
            sample_parsed = sample.str.to_datetime(format=fmt, strict=False)
        except pl.exceptions.ComputeError:
            continue
        if sample_parsed.null_count() > 0:
            continue  # サンプル内に書式不一致があれば別の書式を試す
        parsed = non_null.str.to_datetime(format=fmt, strict=False)
        success_rate = parsed.drop_nulls().len() / non_null.len()
        if success_rate >= DATETIME_PARSE_SUCCESS_THRESHOLD:
            return fmt
    return None


def _parse_datetime_like_columns(df: pl.DataFrame) -> pl.DataFrame:
    """文字列列のうち日時とみなせるものをDatetime型に変換する。

    Args:
        df: 変換対象のDataFrame。

    Returns:
        日時らしき文字列列をDatetime型に変換したDataFrame。
    """
    for col in df.columns:
        if df.schema[col] != pl.Utf8:
            continue
        non_null = df[col].drop_nulls()
        if non_null.len() == 0:
            continue
        fmt = _find_datetime_format(non_null)
        if fmt is None:
            continue
        df = df.with_columns(pl.col(col).str.to_datetime(format=fmt, strict=False).alias(col))
    return df


def read_csv_auto(path: Path) -> pl.DataFrame:
    """CSVファイルをエンコーディング自動判定つきで読み込む。

    UTF-8で読めない場合はCP932（Shift-JIS）として扱う。日時らしき文字列列は
    自動的にDatetime型へ変換する。

    Args:
        path: 読み込むCSVファイルのパス。

    Returns:
        読み込んだDataFrame。
    """
    encoding = detect_encoding(path)
    file_size = path.stat().st_size

    if encoding == "utf-8":
        df = pl.read_csv(
            path, encoding="utf8", infer_schema_length=None, null_values=NULL_VALUE_MARKERS
        )
    elif file_size <= LARGE_FILE_THRESHOLD_BYTES:
        text = path.read_bytes().decode("cp932", errors="replace")
        df = pl.read_csv(
            io.StringIO(text), infer_schema_length=None, null_values=NULL_VALUE_MARKERS
        )
    else:
        with path.open("rb") as f:
            header_line = f.readline()
        header = header_line.decode("cp932", errors="replace").rstrip("\r\n").split(",")
        if _is_ascii_only_from(path, skip_bytes=len(header_line)):
            df = pl.read_csv(
                path,
                has_header=False,
                skip_rows=1,
                new_columns=header,
                encoding="utf8-lossy",
                infer_schema_length=None,
                null_values=NULL_VALUE_MARKERS,
            )
        else:
            text = path.read_bytes().decode("cp932", errors="replace")
            df = pl.read_csv(
                io.StringIO(text), infer_schema_length=None, null_values=NULL_VALUE_MARKERS
            )

    return _parse_datetime_like_columns(df)
