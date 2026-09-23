"""CSVファイルのデータ品質確認（データフレーム全体・カラム単位）を行うユーティリティ。

チェック項目は以下の記事で紹介されているEDA（探索的データ分析）の観点を参考にしている。
https://medium.com/epfl-extension-school/advanced-exploratory-data-analysis-eda-with-python-536fa83c578a

- データフレーム全体: 行数・列数、重複行、欠損値の全体像、記録誤り・外れ値の視覚的確認
- カラム単位: 型・欠損率・ユニーク数、数値列の記述統計と歪度、カテゴリ列の上位出現値、
  数値列間の相関

前提・制約:
    本モジュールが対象とするCSVはUTF-8またはCP932（Shift-JIS）のいずれかで
    エンコードされていることを前提とする。30MBを超える大容量ファイルのうち、
    ヘッダ行を除くデータ本体が完全にASCII文字のみで構成されている場合に限り、
    全文デコードを行わずlazy scanで読み込む（本プロジェクトの `wind_*.csv` が該当）。
    大容量かつデータ本体に非ASCII文字を含むファイルは全文デコードするため、
    メモリ使用量が大きくなる点に注意すること。
    観測データの欠測記号として `*`（アスタリスク）が使われている列があることを
    実データで確認したため、これをnull値として扱う（`wind_*.csv` の風速・風向列が該当）。
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from analysis_project.paths import sanitize_filename_component


def ensure_japanese_font() -> None:
    """日本語フォントをmatplotlibに設定する。

    japanize-matplotlib は distutils 依存のためPython 3.12（setuptools未導入環境）では
    importできないので使用せず、Windowsに標準搭載されている日本語フォントを直接指定する。
    `sns.set_theme()` はフォント設定を含むrcParamsを上書きするため、各描画関数の先頭で
    都度呼び出して確実に日本語フォントが有効な状態で描画する。
    """
    plt.rcParams["font.family"] = "Meiryo"
    plt.rcParams["axes.unicode_minus"] = False  # Meiryoではマイナス記号が文字化けするため


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


def dataframe_overview(df: pl.DataFrame, dataset_name: str, source_path: str) -> pl.DataFrame:
    """データフレーム全体の品質サマリを1行のテーブルとして作成する。

    Args:
        df: 対象のDataFrame。
        dataset_name: 出力ファイル名などに用いるデータセット識別名。
        source_path: 元CSVファイルの `data/raw` からの相対パス（記録用）。

    Returns:
        行数・列数・重複行数・欠損セル数などを含む1行のDataFrame。
    """
    n_rows = df.height
    n_cols = df.width
    n_duplicated_rows = (n_rows - df.unique().height) if n_rows > 0 else 0
    total_cells = n_rows * n_cols
    total_missing = int(df.null_count().sum_horizontal().item()) if n_cols > 0 else 0
    n_cols_with_missing = sum(1 for c in df.columns if df[c].null_count() > 0)

    return pl.DataFrame(
        {
            "dataset": [dataset_name],
            "source_path": [source_path],
            "n_rows": [n_rows],
            "n_columns": [n_cols],
            "n_duplicated_rows": [n_duplicated_rows],
            "duplicated_row_rate": [n_duplicated_rows / n_rows if n_rows else None],
            "n_columns_with_missing": [n_cols_with_missing],
            "total_missing_cells": [total_missing],
            "missing_cell_rate": [total_missing / total_cells if total_cells else None],
            "estimated_memory_mb": [df.estimated_size("mb")],
        }
    )


def column_overview(df: pl.DataFrame, dataset_name: str) -> pl.DataFrame:
    """カラム単位の品質サマリテーブルを作成する。

    全列について型・欠損率・ユニーク数・最頻値とその出現率を求め、数値列については
    記述統計（平均・標準偏差・四分位・最小最大）と歪度を追加で求める。

    Args:
        df: 対象のDataFrame。
        dataset_name: 出力ファイル名などに用いるデータセット識別名。

    Returns:
        列ごとに1行を持つ品質サマリのDataFrame。
    """
    n_rows = df.height
    rows: list[dict[str, object]] = []

    for col in df.columns:
        s = df[col]
        dtype = s.dtype
        is_numeric = dtype.is_numeric()
        n_missing = s.null_count()
        n_unique = s.n_unique()
        non_null = s.drop_nulls()

        mode_value: object = None
        mode_rate: float | None = None
        if non_null.len() > 0:
            vc = non_null.value_counts(sort=True)
            count_col = "count"
            mode_value = vc[col][0]
            mode_rate = vc[count_col][0] / non_null.len()

        row: dict[str, object] = {
            "dataset": dataset_name,
            "column": col,
            "dtype": str(dtype),
            "is_numeric": is_numeric,
            "n_rows": n_rows,
            "n_missing": n_missing,
            "missing_rate": n_missing / n_rows if n_rows else None,
            "n_unique": n_unique,
            "unique_rate": n_unique / n_rows if n_rows else None,
            "mode": str(mode_value) if mode_value is not None else None,
            "mode_rate": mode_rate,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "skewness": None,
        }

        if is_numeric and non_null.len() > 0:
            row.update(
                {
                    "mean": non_null.mean(),
                    "std": non_null.std(),
                    "min": non_null.min(),
                    "p25": non_null.quantile(0.25),
                    "median": non_null.median(),
                    "p75": non_null.quantile(0.75),
                    "max": non_null.max(),
                    "skewness": non_null.skew(),
                }
            )

        rows.append(row)

    return pl.DataFrame(rows)


def add_caption(fig: plt.Figure, text: str) -> None:
    """図の下部にキャプション（対象の説明文）を追加する。

    Args:
        fig: 対象のFigure。
        text: キャプション文字列。
    """
    fig.text(0.01, -0.02, text, ha="left", va="top", fontsize=9, wrap=True)


def plot_missing_overview(
    df: pl.DataFrame, dataset_name: str, n_rows: int, output_path: Path
) -> bool:
    """カラム別の欠損率を棒グラフとして保存する（データフレーム全体の欠損値確認）。

    Args:
        df: 対象のDataFrame。
        dataset_name: 図のタイトル・キャプションに使うデータセット識別名。
        n_rows: 元データの行数。
        output_path: 保存先のPNGパス。

    Returns:
        図を保存した場合は True、欠損が1件もなく図を作成しなかった場合は False。
    """
    if n_rows == 0:
        return False

    missing_rates = [df[c].null_count() / n_rows for c in df.columns]
    if sum(missing_rates) == 0:
        return False

    order = sorted(zip(df.columns, missing_rates, strict=True), key=lambda t: t[1], reverse=True)
    columns_sorted = [c for c, _ in order]
    rates_sorted = [r for _, r in order]

    ensure_japanese_font()
    fig, ax = plt.subplots(
        figsize=(10, max(4, 0.35 * len(columns_sorted))), constrained_layout=True
    )
    ax.barh(columns_sorted, rates_sorted, color=sns.color_palette("muted")[0])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("欠損率")
    ax.set_ylabel("カラム")
    ax.set_title(f"{dataset_name}: カラム別欠損率")
    add_caption(fig, f"対象: {dataset_name} (n={n_rows:,}行) — 各カラムの欠損値の割合を示す。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_numeric_histograms(
    df: pl.DataFrame, dataset_name: str, n_rows: int, output_path: Path
) -> bool:
    """数値カラムのヒストグラムをグリッド形式で保存する。

    Args:
        df: 対象のDataFrame。
        dataset_name: 図のタイトル・キャプションに使うデータセット識別名。
        n_rows: 元データの行数。
        output_path: 保存先のPNGパス。

    Returns:
        図を保存した場合は True、数値カラムが存在せず作成しなかった場合は False。
    """
    numeric_cols = [c for c in df.columns if df.schema[c].is_numeric()]
    if not numeric_cols:
        return False

    ncols = min(4, len(numeric_cols))
    nrows = -(-len(numeric_cols) // ncols)
    ensure_japanese_font()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), constrained_layout=True, squeeze=False
    )

    for i, col in enumerate(numeric_cols):
        ax = axes[i // ncols][i % ncols]
        values = df[col].drop_nulls().to_numpy()
        if values.size == 0:
            ax.set_visible(False)
            continue
        ax.hist(values, bins=50, color=sns.color_palette("muted")[0])
        ax.set_title(col, fontsize=11)
        ax.set_ylabel("度数")

    for j in range(len(numeric_cols), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"{dataset_name}: 数値カラムのヒストグラム")
    add_caption(
        fig,
        f"対象: {dataset_name} (n={n_rows:,}行) — 各数値カラムの値分布（ヒストグラム、50 bins）。",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_categorical_top_values(
    df: pl.DataFrame,
    dataset_name: str,
    n_rows: int,
    output_path: Path,
    top_n: int = 15,
    max_unique_rate: float = 0.5,
) -> bool:
    """非数値カラムの上位出現値を棒グラフのグリッド形式で保存する。

    ユニーク率が `max_unique_rate` を超えるカラム（IDなどほぼ一意な列）は、
    棒グラフとして意味を持たないため対象から除外する。

    Args:
        df: 対象のDataFrame。
        dataset_name: 図のタイトル・キャプションに使うデータセット識別名。
        n_rows: 元データの行数。
        output_path: 保存先のPNGパス。
        top_n: 表示する上位カテゴリ数。
        max_unique_rate: カテゴリ列とみなす最大ユニーク率。

    Returns:
        図を保存した場合は True、対象カラムが存在せず作成しなかった場合は False。
    """
    if n_rows == 0:
        return False

    cat_cols = []
    for c in df.columns:
        if df.schema[c].is_numeric() or df.schema[c] in (pl.Datetime, pl.Date):
            continue
        n_unique = df[c].n_unique()
        if n_unique / n_rows > max_unique_rate:
            continue
        cat_cols.append(c)
    if not cat_cols:
        return False

    ncols = min(3, len(cat_cols))
    nrows = -(-len(cat_cols) // ncols)
    ensure_japanese_font()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), constrained_layout=True, squeeze=False
    )

    for i, col in enumerate(cat_cols):
        ax = axes[i // ncols][i % ncols]
        vc = df[col].drop_nulls().value_counts(sort=True).head(top_n)
        labels = [str(v) for v in vc[col].to_list()]
        counts = vc["count"].to_list()
        ax.barh(labels, counts, color=sns.color_palette("muted")[1])
        ax.invert_yaxis()
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("件数")

    for j in range(len(cat_cols), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"{dataset_name}: カテゴリ列の上位出現値（上位{top_n}件）")
    add_caption(
        fig,
        f"対象: {dataset_name} (n={n_rows:,}行) — 各カテゴリ列で出現頻度が高い上位{top_n}件"
        f"（ユニーク率{max_unique_rate:.0%}超の列は除外）。",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def complete_case_correlation(
    df: pl.DataFrame, numeric_cols: list[str]
) -> tuple[pl.DataFrame, int]:
    """数値カラムのいずれかに欠損値がある行を除外してピアソン相関係数を算出する。

    polarsの `DataFrame.corr()` は、対象カラムに欠損値を含む行が1件でもあると
    相関行列全体がNaNになってしまうため、事前にリストワイズ削除（対象カラムの
    いずれかがnullの行をまとめて除外）してから算出する。

    Args:
        df: 対象のDataFrame。
        numeric_cols: 相関係数を算出する数値カラム名のリスト（2列以上）。

    Returns:
        (相関行列のDataFrame, 欠損値により除外した行数) のタプル。
    """
    numeric_df = df.select(numeric_cols)
    complete_df = numeric_df.drop_nulls()
    n_excluded = numeric_df.height - complete_df.height
    return complete_df.corr(), n_excluded


def plot_correlation_heatmap(
    df: pl.DataFrame, dataset_name: str, n_rows: int, output_path: Path
) -> bool:
    """数値カラム間のピアソン相関係数をヒートマップとして保存する。

    数値カラムのいずれかに欠損値がある行は、相関係数の算出前に除外する
    （polarsの `corr()` は欠損値を含む行があると全体がNaNになるため）。
    除外した場合は、除外した行数をキャプションに明記する。

    Args:
        df: 対象のDataFrame。
        dataset_name: 図のタイトル・キャプションに使うデータセット識別名。
        n_rows: 元データの行数。
        output_path: 保存先のPNGパス。

    Returns:
        図を保存した場合は True、数値カラムが2列未満、または欠損値を除いた結果
        相関係数を算出できるサンプルが2件未満だった場合は False。
    """
    numeric_cols = [c for c in df.columns if df.schema[c].is_numeric()]
    if len(numeric_cols) < 2:
        return False

    corr, n_excluded = complete_case_correlation(df, numeric_cols)
    n_valid = n_rows - n_excluded
    if n_valid < 2:
        return False

    ensure_japanese_font()
    fig, ax = plt.subplots(
        figsize=(1.2 * len(numeric_cols) + 2, 1.0 * len(numeric_cols) + 2),
        constrained_layout=True,
    )
    sns.heatmap(
        corr.to_numpy(),
        xticklabels=numeric_cols,
        yticklabels=numeric_cols,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        vmin=-1,
        vmax=1,
        ax=ax,
    )
    ax.set_title(f"{dataset_name}: 数値カラム間の相関係数（ピアソン）")
    caption = f"対象: {dataset_name} (n={n_rows:,}行) — 数値カラム間のピアソン相関係数。"
    if n_excluded > 0:
        caption += f" 欠損値を含む{n_excluded:,}行を除外して算出（有効サンプル数: {n_valid:,}）。"
    add_caption(fig, caption)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_record_pattern(
    df: pl.DataFrame, dataset_name: str, n_rows: int, output_path: Path, max_points: int = 5000
) -> bool:
    """数値カラムを行の並び順に沿って間引きプロットし、記録誤りや外れ値を目視確認する。

    日時型の列が存在する場合はそれをx軸に、存在しない場合は行インデックスをx軸に用いる。

    Args:
        df: 対象のDataFrame。
        dataset_name: 図のタイトル・キャプションに使うデータセット識別名。
        n_rows: 元データの行数。
        output_path: 保存先のPNGパス。
        max_points: 1カラムあたりの最大プロット点数（間引き後）。

    Returns:
        図を保存した場合は True、数値カラムが存在せず作成しなかった場合は False。
    """
    numeric_cols = [c for c in df.columns if df.schema[c].is_numeric()]
    if not numeric_cols or n_rows == 0:
        return False

    datetime_cols = [c for c in df.columns if df.schema[c] in (pl.Datetime, pl.Date)]
    x_col = datetime_cols[0] if datetime_cols else None

    step = max(1, n_rows // max_points)
    sampled = df.with_row_index("_row_idx").filter(pl.col("_row_idx") % step == 0)
    x = sampled[x_col].to_numpy() if x_col else sampled["_row_idx"].to_numpy()

    ncols = min(4, len(numeric_cols))
    nrows = -(-len(numeric_cols) // ncols)
    ensure_japanese_font()
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), constrained_layout=True, squeeze=False
    )

    for i, col in enumerate(numeric_cols):
        ax = axes[i // ncols][i % ncols]
        y = sampled[col].to_numpy()
        ax.plot(x, y, lw=0, marker=".", markersize=2, color=sns.color_palette("muted")[2])
        ax.set_title(col, fontsize=11)
        if x_col:
            ax.tick_params(axis="x", rotation=30)

    for j in range(len(numeric_cols), nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"{dataset_name}: 数値カラムの記録パターン（記録誤り・外れ値の目視確認用）")
    x_axis_desc = f"x軸: {x_col}" if x_col else "x軸: 行インデックス"
    add_caption(
        fig,
        f"対象: {dataset_name} (n={n_rows:,}行を最大{max_points:,}点に間引いて表示, {x_axis_desc}) "
        "— 行の並び順に沿った値の推移。",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def chunk_numeric_columns(columns: list[str], max_per_chunk: int = 3) -> list[list[str]]:
    """カラム名のリストを、先頭から `max_per_chunk` 列ずつのグループに分割する。

    Args:
        columns: 分割対象のカラム名リスト。
        max_per_chunk: 1グループに含める最大カラム数。

    Returns:
        カラム名のリストのリスト（各要素が1グループ）。`columns` が空なら空リスト。
    """
    return [columns[i : i + max_per_chunk] for i in range(0, len(columns), max_per_chunk)]


def plot_scatter_matrix(
    df: pl.DataFrame,
    row_columns: list[str],
    col_columns: list[str],
    dataset_name: str,
    n_rows: int,
    output_path: Path,
    alpha: float = 0.3,
) -> bool:
    """指定した行カラム×列カラムの組み合わせについて散布図行列を保存する。

    セルの行カラムと列カラムが同じ場合（`row_columns` と `col_columns` に共通のカラムが
    使われる場合）のみ、そのセルはヒストグラム（該当カラムの分布）にする。それ以外は
    散布図とし、`alpha` で点を半透明にしてデータの重なりが濃淡で表現されるようにする。
    各セルは、そのセルで使う2列（ヒストグラムの場合は1列）に欠損値を含む行を除いて描画する。

    `row_columns` と `col_columns` が同じカラム集合であれば正方形の行列（対角がヒストグラム）
    になり、互いに素なカラム集合であれば長方形の行列（対角なし、全セル散布図）になる。

    Args:
        df: 対象のDataFrame。
        row_columns: 行（縦軸）に配置する数値カラム名。
        col_columns: 列（横軸）に配置する数値カラム名。
        dataset_name: 図のタイトル・キャプションに使うデータセット識別名。
        n_rows: 元データの行数。
        output_path: 保存先のPNGパス。
        alpha: 散布図の点の透明度（0に近いほど透明。重なりを濃淡で表現するために使う）。

    Returns:
        図を保存した場合は True、行または列のカラムが空で作成しなかった場合は False。
    """
    if not row_columns or not col_columns:
        return False

    n_row = len(row_columns)
    n_col = len(col_columns)
    ensure_japanese_font()
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(3.2 * n_col, 3.2 * n_row), constrained_layout=True, squeeze=False
    )
    color = sns.color_palette("muted")[0]

    for i, row_col in enumerate(row_columns):
        for j, col_col in enumerate(col_columns):
            ax = axes[i][j]
            if row_col == col_col:
                values = df[row_col].drop_nulls().to_numpy()
                if values.size > 0:
                    ax.hist(values, bins=30, color=color)
            else:
                pair = df.select([col_col, row_col]).drop_nulls()
                if pair.height > 0:
                    ax.scatter(
                        pair[col_col],
                        pair[row_col],
                        alpha=alpha,
                        s=10,
                        color=color,
                        edgecolors="none",
                    )
            if i == n_row - 1:
                ax.set_xlabel(col_col, fontsize=9)
            if j == 0:
                ax.set_ylabel(row_col, fontsize=9)

    row_label = " / ".join(row_columns)
    col_label = " / ".join(col_columns)
    title_cols = row_label if row_columns == col_columns else f"{row_label} × {col_label}"
    fig.suptitle(f"{dataset_name}: 散布図行列（{title_cols}）")
    add_caption(
        fig,
        f"対象: {dataset_name} (n={n_rows:,}行) — 縦横で同じカラムのセルはヒストグラム、"
        f"それ以外は散布図（点の透明度alpha={alpha}で重なりを濃淡表現）。"
        "各散布図は該当2列の欠損値を除いて描画。",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_scatter_matrices(
    df: pl.DataFrame,
    dataset_name: str,
    n_rows: int,
    figures_dir: Path,
    max_cols_per_image: int = 3,
    alpha: float = 0.3,
) -> list[Path]:
    """数値カラムを `max_cols_per_image` 列ずつのグループに分割し、
    グループ同士のすべての組み合わせについて散布図行列を保存する。

    グループが2つ以上ある場合、同一グループ同士（正方形、対角がヒストグラム）だけでなく
    異なるグループ同士（長方形、全セル散布図）の組み合わせも作成することで、
    全カラムの総当たりの組み合わせに抜け漏れが出ないようにする。グループ(i, j)と
    グループ(j, i)は縦横を入れ替えただけの同じ組み合わせを表すため、i <= j の場合のみ
    作成する。

    Args:
        df: 対象のDataFrame。
        dataset_name: 出力ファイル名・図のタイトルに使うデータセット識別名。
        n_rows: 元データの行数。
        figures_dir: 図の出力先ディレクトリ。
        max_cols_per_image: 1画像に含める最大カラム数（縦・横それぞれ）。
        alpha: 散布図の点の透明度。

    Returns:
        作成した画像ファイルのパスのリスト。
    """
    numeric_cols = [c for c in df.columns if df.schema[c].is_numeric()]
    chunks = chunk_numeric_columns(numeric_cols, max_cols_per_image)

    created: list[Path] = []
    for i, row_chunk in enumerate(chunks):
        for j, col_chunk in enumerate(chunks):
            if i > j:
                continue  # (j, i) は (i, j) の縦横を入れ替えただけなので重複作成しない
            if i == j and len(row_chunk) < 2:
                continue  # 1列だけの自己組み合わせはヒストグラム1枚のみになるため対象外

            row_suffix = sanitize_filename_component("_".join(row_chunk))
            if i == j:
                name = f"{dataset_name}__scatter_matrix__{row_suffix}.png"
            else:
                col_suffix = sanitize_filename_component("_".join(col_chunk))
                name = f"{dataset_name}__scatter_matrix__{row_suffix}__x__{col_suffix}.png"
            output_path = figures_dir / name

            if plot_scatter_matrix(
                df, row_chunk, col_chunk, dataset_name, n_rows, output_path, alpha=alpha
            ):
                created.append(output_path)
    return created


def make_dataset_name(path: Path, raw_dir: Path) -> str:
    """CSVファイルのパスから出力ファイル名に使うデータセット識別名を作る。

    サブディレクトリを含む相対パスをアンダースコアで連結することで、
    異なるディレクトリに同名のファイルがあっても衝突しないようにする。

    Args:
        path: 対象CSVファイルのパス。
        raw_dir: `data/raw` のパス。

    Returns:
        例: `data/raw/wind_0/下条川.csv` -> `wind_0_下条川`。
    """
    rel = path.relative_to(raw_dir).with_suffix("")
    return "_".join(rel.parts)


def run_quality_checks(
    df: pl.DataFrame,
    dataset_name: str,
    source_path: str,
    figures_dir: Path,
    tables_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """読み込み済みのDataFrameに対してデータ品質確認を実施し、表と図を保存する。

    大容量ファイルを複数回読み込まずに済むよう、CSVの読み込みは呼び出し側
    （`read_csv_auto`）で行い、結果のDataFrameを受け取る設計にしている。

    Args:
        df: `read_csv_auto` で読み込み済みのDataFrame。
        dataset_name: 出力ファイル名・図のタイトルに使うデータセット識別名。
        source_path: 元CSVファイルの `data/raw` からの相対パス（記録用）。
        figures_dir: 図の出力先ディレクトリ。
        tables_dir: 表の出力先ディレクトリ。

    Returns:
        (データフレーム全体サマリの1行DataFrame, カラム単位サマリのN行DataFrame)。
    """
    n_rows = df.height

    df_overview = dataframe_overview(df, dataset_name, source_path)
    col_overview = column_overview(df, dataset_name)

    tables_dir.mkdir(parents=True, exist_ok=True)
    df_overview.write_csv(tables_dir / f"{dataset_name}__dataframe_overview.csv")
    col_overview.write_csv(tables_dir / f"{dataset_name}__column_overview.csv")

    plot_missing_overview(
        df, dataset_name, n_rows, figures_dir / f"{dataset_name}__missing_overview.png"
    )
    plot_numeric_histograms(
        df, dataset_name, n_rows, figures_dir / f"{dataset_name}__numeric_histograms.png"
    )
    plot_categorical_top_values(
        df, dataset_name, n_rows, figures_dir / f"{dataset_name}__categorical_top_values.png"
    )
    plot_correlation_heatmap(
        df, dataset_name, n_rows, figures_dir / f"{dataset_name}__correlation_heatmap.png"
    )
    plot_record_pattern(
        df, dataset_name, n_rows, figures_dir / f"{dataset_name}__record_pattern.png"
    )
    plot_scatter_matrices(df, dataset_name, n_rows, figures_dir)

    return df_overview, col_overview
