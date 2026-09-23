"""時系列データを含むCSVに対する時系列EDA（探索的データ分析）ユーティリティ。

対象系列（原系列から作る6種類の変換系列）:
    - raw: 原系列
    - log: 対数系列（0以下の値はnullとして扱う）
    - diff: 差分系列（1階差分）
    - log_diff: 対数差分系列（対数系列の1階差分）
    - seasonal_diff: 季節差分系列（デフォルト7点、設定可能）
    - seasonal_log_diff: 季節対数差分系列（対数系列の季節差分、デフォルト7点、設定可能）

対象グラフ（上記6系列それぞれについて作成）:
    - 生値と移動平均（デフォルト5点、設定可能）
    - 自己相関係数（ACF）のコレログラム
    - 偏自己相関係数（PACF）のコレログラム

前提・制約:
    - 日時型（Datetime/Date）の列を持たないDataFrameは対象外としてスキップする。
    - 日時列に重複がある場合（例: 複数地点や複数系列が1ファイルに混在するパネルデータ）、
      日時列と組み合わせて行を一意に識別できる非数値列（グループ列）を1列だけ自動探索し、
      見つかった場合はグループごとに独立した時系列として扱う。該当する単一列が見つからない
      場合は、誤ったコレログラムを出さないよう当該データセットの時系列分析はスキップする。
    - 自己相関係数・偏自己相関係数の計算コストを抑えるため、非常に長い系列は直近
      `max_acf_points` 点のみを用いて計算する（データ全体の並び順・間隔はそのまま利用し、
      間引きはしない）。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf

from analysis_project.csv_quality import add_caption, ensure_japanese_font

# ACF/PACF計算に用いる最大点数。これを超える場合は直近この件数のみを使用する。
DEFAULT_MAX_ACF_POINTS = 20_000

# 変換系列のキーと日本語ラベル（グラフの並び順・タイトルに使用）。
SERIES_LABELS: dict[str, str] = {
    "raw": "原系列",
    "log": "対数系列",
    "diff": "差分系列",
    "log_diff": "対数差分系列",
    "seasonal_diff": "季節差分系列",
    "seasonal_log_diff": "季節対数差分系列",
}
_SERIES_GRID_ORDER = list(SERIES_LABELS)


_FILENAME_UNSAFE_CHARS = str.maketrans({c: "_" for c in '\\/:*?"<>|'})


def sanitize_filename_component(text: str) -> str:
    """ファイル名に使えない文字（Windowsで禁止されている記号）をアンダースコアに置き換える。

    グループ列の値（例: 時刻文字列 "06:41:00"）がそのままファイル名の一部になる場合が
    あるため、保存直前にこの関数でサニタイズする。図のタイトル・キャプションに使う
    文字列は元の値のまま（サニタイズ前）を使う。

    Args:
        text: サニタイズ対象の文字列。

    Returns:
        `\\ / : * ? " < > |` をアンダースコアに置き換えた文字列。
    """
    return text.translate(_FILENAME_UNSAFE_CHARS)


def find_datetime_column(df: pl.DataFrame) -> str | None:
    """DataFrameから日時型（Datetime/Date）の列を探す。

    Args:
        df: 対象のDataFrame。

    Returns:
        最初に見つかった日時型列の名前。存在しなければ None。
    """
    for c in df.columns:
        if df.schema[c] in (pl.Datetime, pl.Date):
            return c
    return None


def find_numeric_columns(df: pl.DataFrame, exclude: tuple[str, ...] = ()) -> list[str]:
    """DataFrameから数値型の列名一覧を取得する。

    Args:
        df: 対象のDataFrame。
        exclude: 除外する列名。

    Returns:
        数値型の列名リスト（元の列順）。
    """
    return [c for c in df.columns if c not in exclude and df.schema[c].is_numeric()]


def find_grouping_column(df: pl.DataFrame, datetime_col: str) -> str | None:
    """日時列の重複を解消できる単一の分類列を探す。

    複数地点・複数系列が1つのファイルに混在するパネルデータ形式（例: 気象データの地点別
    観測値）かどうかを判定するために使う。日時列に重複がない場合は None を返す
    （＝グループ化不要で単一系列として扱ってよい）。

    Args:
        df: 対象のDataFrame。
        datetime_col: 日時型の列名。

    Returns:
        (datetime_col, 当該列) の組み合わせが行数と一致する最初の非数値列名。
        日時列に重複がない場合、または該当する単一列が見つからない場合は None。
    """
    n_rows = df.height
    if df[datetime_col].n_unique() == n_rows:
        return None
    for c in df.columns:
        if c == datetime_col or df.schema[c].is_numeric():
            continue
        if df.select([datetime_col, c]).n_unique() == n_rows:
            return c
    return None


def to_log_series(values: pl.Series) -> pl.Series:
    """0以下の値をnullとしたうえで自然対数変換を行う。

    対数は正の値でのみ定義されるため、0以下の値は欠損として扱う。

    Args:
        values: 変換対象の数値Series。

    Returns:
        対数変換後のSeries（0以下だった要素はnull）。元の列名を維持する。
    """
    name = values.name
    frame = values.to_frame("v")
    return frame.select(
        pl.when(pl.col("v") > 0).then(pl.col("v")).otherwise(None).log().alias(name)
    ).to_series()


def build_transformed_series(values: pl.Series, seasonal_period: int = 7) -> dict[str, pl.Series]:
    """原系列から6種類の変換系列を作成する。

    Args:
        values: 原系列の数値Series。
        seasonal_period: 季節差分・季節対数差分の周期（ラグ数）。

    Returns:
        `SERIES_LABELS` のキーに対応する変換系列の辞書。
    """
    log_values = to_log_series(values)
    return {
        "raw": values,
        "log": log_values,
        "diff": values.diff(),
        "log_diff": log_values.diff(),
        "seasonal_diff": values.diff(seasonal_period),
        "seasonal_log_diff": log_values.diff(seasonal_period),
    }


def moving_average(values: pl.Series, window: int = 5) -> pl.Series:
    """単純移動平均を計算する。

    Args:
        values: 対象の数値Series。
        window: 移動平均の窓幅（点数）。

    Returns:
        移動平均のSeries（先頭 `window - 1` 件はnull）。
    """
    return values.rolling_mean(window_size=window)


def _prepare_for_acf(values: pl.Series, max_points: int) -> np.ndarray | None:
    """ACF/PACF計算用に欠損除去・末尾切り詰めを行いnumpy配列に変換する。

    Args:
        values: 対象のSeries。
        max_points: 計算に用いる最大点数。超える場合は直近 `max_points` 点のみを使う。

    Returns:
        計算に十分なデータ（10点以上）があればnumpy配列、なければ None。
    """
    non_null = values.drop_nulls()
    if non_null.len() < 10:
        return None
    if non_null.len() > max_points:
        non_null = non_null.tail(max_points)
    return non_null.to_numpy()


def compute_acf(
    values: pl.Series, nlags: int = 40, max_points: int = DEFAULT_MAX_ACF_POINTS
) -> np.ndarray | None:
    """自己相関係数（ACF）をラグ0からnlagsまで計算する。

    Args:
        values: 対象のSeries。
        nlags: 計算するラグ数の上限。
        max_points: 計算に用いる最大点数（性能確保のため直近max_points点に切り詰める）。

    Returns:
        長さ `nlags + 1`（データが少ない場合はそれ以下）のACF配列。
        データ不足（10点未満）の場合は None。
    """
    arr = _prepare_for_acf(values, max_points)
    if arr is None:
        return None
    effective_nlags = min(nlags, arr.size - 1)
    if effective_nlags < 1:
        return None
    return np.asarray(acf(arr, nlags=effective_nlags, fft=True))


def compute_pacf(
    values: pl.Series, nlags: int = 40, max_points: int = DEFAULT_MAX_ACF_POINTS
) -> np.ndarray | None:
    """偏自己相関係数（PACF）をラグ0からnlagsまで計算する。

    Args:
        values: 対象のSeries。
        nlags: 計算するラグ数の上限。
        max_points: 計算に用いる最大点数（性能確保のため直近max_points点に切り詰める）。

    Returns:
        長さ `nlags + 1`（データが少ない場合はそれ以下）のPACF配列。
        データ不足の場合は None。
    """
    arr = _prepare_for_acf(values, max_points)
    if arr is None:
        return None
    # PACF(Yule-Walker)はnlagsがサンプルサイズの半分未満である必要がある
    effective_nlags = min(nlags, arr.size // 2 - 1)
    if effective_nlags < 1:
        return None
    return np.asarray(pacf(arr, nlags=effective_nlags))


def break_line_at_gaps(x: np.ndarray, y: np.ndarray, gap_factor: float = 10.0) -> np.ndarray:
    """時間軸の間隔が異常に大きい箇所でyをnanにし、折れ線グラフを分断する。

    欠測期間（記録が長期間存在しない区間）をまたいで直線で結んでしまうと、
    あたかも連続した観測があるかのように誤解を招くため、典型的な間隔の
    `gap_factor` 倍を超える空白が見つかった場合、その直後の点をnanにして線を切る。

    Args:
        x: 時間軸の値（`numpy.datetime64` 配列、または単調増加の数値配列）。
        y: `x` と対応するプロット対象の値。
        gap_factor: 「異常に大きい間隔」とみなす、典型的な間隔（中央値）に対する倍率。

    Returns:
        欠測期間の直後をnanに置き換えた `y` のコピー。
    """
    y = y.astype(float).copy()
    if len(x) < 3:
        return y
    if np.issubdtype(x.dtype, np.datetime64):
        diffs = np.diff(x).astype("timedelta64[s]").astype(float)
    else:
        diffs = np.diff(x).astype(float)
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return y
    median_diff = np.median(positive_diffs)
    gap_indices = np.where(diffs > median_diff * gap_factor)[0] + 1
    y[gap_indices] = np.nan
    return y


def _plot_value_with_ma_ax(
    ax: plt.Axes, dt: pl.Series, values: pl.Series, window: int, max_points: int
) -> None:
    """1系列について生値と移動平均をAxesに描画する。"""
    ma = moving_average(values, window)
    frame = pl.DataFrame({"dt": dt, "value": values, "ma": ma}).with_row_index("_row_idx")
    step = max(1, frame.height // max_points)
    sampled = frame.filter(pl.col("_row_idx") % step == 0)
    x = sampled["dt"].to_numpy()
    value_y = break_line_at_gaps(x, sampled["value"].to_numpy())
    ma_y = break_line_at_gaps(x, sampled["ma"].to_numpy())
    colors = sns.color_palette("muted")
    ax.plot(x, value_y, lw=0.6, alpha=0.5, color=colors[0], label="生値")
    ax.plot(x, ma_y, lw=1.3, color=colors[3], label=f"移動平均({window}点)")
    ax.tick_params(axis="x", rotation=30)


def _plot_empty_ax(ax: plt.Axes) -> None:
    """データ不足を示すプレースホルダーをAxesに描画する。"""
    ax.text(0.5, 0.5, "データ不足", ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _plot_correlogram_ax(ax: plt.Axes, values_arr: np.ndarray | None, n_obs: int) -> None:
    """ACF/PACFの棒グラフ（コレログラム）と95%信頼区間の目安帯をAxesに描画する。"""
    if values_arr is None:
        _plot_empty_ax(ax)
        return
    lags = np.arange(len(values_arr))
    ax.vlines(lags, 0, values_arr, color=sns.color_palette("muted")[2])
    ax.axhline(0, color="black", lw=0.8)
    conf = 1.96 / np.sqrt(n_obs)
    ax.axhspan(-conf, conf, color="gray", alpha=0.2)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("ラグ")


def _new_grid_figure(
    dataset_name: str, column_name: str, title_suffix: str
) -> tuple[plt.Figure, np.ndarray]:
    """2行3列（6種類の変換系列）のグリッド図を作成する。

    この関数が返す図は `constrained_layout` を使わない（プロジェクトの既定は
    `constrained_layout=True` だが、この実行環境ではテキスト描画のたびに
    `constrained_layout` がレイアウトを反復計算するコストが非常に大きく、
    1系列につき最大3枚×6パネルの図を大量生成するこのモジュールでは
    `tight_layout()` に切り替えることで描画時間を約4割削減できることを
    実測で確認したため、この関数に限り例外的に `tight_layout()` を使う）。
    呼び出し側は図の内容を組み立てた後、保存前に `fig.tight_layout(...)` を
    呼び出すこと。
    """
    ensure_japanese_font()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.suptitle(f"{dataset_name} / {column_name}: {title_suffix}")
    return fig, axes


def plot_series_with_moving_average(
    dt: pl.Series,
    series_dict: dict[str, pl.Series],
    dataset_name: str,
    column_name: str,
    output_path: Path,
    moving_average_window: int = 5,
    max_points: int = 5000,
) -> bool:
    """6種類の変換系列について、生値と移動平均を2行3列のグリッドにまとめて保存する。

    Args:
        dt: 時系列の日時Series（`series_dict` の各系列と対応する順序でソート済みのもの）。
        series_dict: `build_transformed_series` が返す6系列の辞書。
        dataset_name: 図のタイトル・キャプションに使うデータセット識別名。
        column_name: 対象の数値カラム名。
        output_path: 保存先のPNGパス。
        moving_average_window: 移動平均の窓幅（点数）。
        max_points: 1系列あたりの最大プロット点数（間引き後）。

    Returns:
        図を保存した場合は True、原系列にデータが1件もなく作成しなかった場合は False。
    """
    if series_dict["raw"].drop_nulls().len() == 0:
        return False

    fig, axes = _new_grid_figure(
        dataset_name, column_name, f"生値と移動平均（{moving_average_window}点）"
    )
    for ax, key in zip(axes.flat, _SERIES_GRID_ORDER, strict=True):
        series = series_dict[key]
        if series.drop_nulls().len() == 0:
            _plot_empty_ax(ax)
        else:
            _plot_value_with_ma_ax(ax, dt, series, moving_average_window, max_points)
        ax.set_title(SERIES_LABELS[key], fontsize=11)
    axes.flat[0].legend(loc="upper left", fontsize=8)

    add_caption(
        fig,
        f"対象: {dataset_name} / {column_name} (n={dt.len():,}行) — "
        f"6種類の変換系列の生値（薄色）と移動平均（太線、{moving_average_window}点）。",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_acf_correlogram(
    series_dict: dict[str, pl.Series],
    dataset_name: str,
    column_name: str,
    output_path: Path,
    nlags: int = 40,
    max_points: int = DEFAULT_MAX_ACF_POINTS,
) -> bool:
    """6種類の変換系列について、自己相関係数（ACF）のコレログラムをグリッドにまとめて保存する。

    Args:
        series_dict: `build_transformed_series` が返す6系列の辞書。
        dataset_name: 図のタイトル・キャプションに使うデータセット識別名。
        column_name: 対象の数値カラム名。
        output_path: 保存先のPNGパス。
        nlags: 計算・表示するラグ数の上限。
        max_points: ACF計算に用いる最大点数（直近max_points点に切り詰める）。

    Returns:
        図を保存した場合は True、原系列にデータが1件もなく作成しなかった場合は False。
    """
    if series_dict["raw"].drop_nulls().len() == 0:
        return False

    fig, axes = _new_grid_figure(dataset_name, column_name, "自己相関係数（ACF）のコレログラム")
    for ax, key in zip(axes.flat, _SERIES_GRID_ORDER, strict=True):
        series = series_dict[key]
        acf_vals = compute_acf(series, nlags=nlags, max_points=max_points)
        _plot_correlogram_ax(ax, acf_vals, series.drop_nulls().len())
        ax.set_title(SERIES_LABELS[key], fontsize=11)
        ax.set_ylabel("ACF")

    add_caption(
        fig,
        f"対象: {dataset_name} / {column_name} — 6種類の変換系列のACF（ラグ0〜{nlags}、"
        f"直近最大{max_points:,}点を使用）。灰色帯は95%信頼区間の目安。",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def plot_pacf_correlogram(
    series_dict: dict[str, pl.Series],
    dataset_name: str,
    column_name: str,
    output_path: Path,
    nlags: int = 40,
    max_points: int = DEFAULT_MAX_ACF_POINTS,
) -> bool:
    """6種類の変換系列について、偏自己相関係数（PACF）のコレログラムをグリッドにまとめて保存する。

    Args:
        series_dict: `build_transformed_series` が返す6系列の辞書。
        dataset_name: 図のタイトル・キャプションに使うデータセット識別名。
        column_name: 対象の数値カラム名。
        output_path: 保存先のPNGパス。
        nlags: 計算・表示するラグ数の上限。
        max_points: PACF計算に用いる最大点数（直近max_points点に切り詰める）。

    Returns:
        図を保存した場合は True、原系列にデータが1件もなく作成しなかった場合は False。
    """
    if series_dict["raw"].drop_nulls().len() == 0:
        return False

    fig, axes = _new_grid_figure(dataset_name, column_name, "偏自己相関係数（PACF）のコレログラム")
    for ax, key in zip(axes.flat, _SERIES_GRID_ORDER, strict=True):
        series = series_dict[key]
        pacf_vals = compute_pacf(series, nlags=nlags, max_points=max_points)
        _plot_correlogram_ax(ax, pacf_vals, series.drop_nulls().len())
        ax.set_title(SERIES_LABELS[key], fontsize=11)
        ax.set_ylabel("PACF")

    add_caption(
        fig,
        f"対象: {dataset_name} / {column_name} — 6種類の変換系列のPACF（ラグ0〜{nlags}、"
        f"直近最大{max_points:,}点を使用）。灰色帯は95%信頼区間の目安。",
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def run_time_series_checks(
    df: pl.DataFrame,
    dataset_name: str,
    figures_dir: Path,
    moving_average_window: int = 5,
    seasonal_period: int = 7,
    nlags: int = 40,
    max_plot_points: int = 5000,
    max_acf_points: int = DEFAULT_MAX_ACF_POINTS,
) -> list[str]:
    """時系列データを含むDataFrameに対し、数値列ごとに時系列診断図一式を作成する。

    日時型の列が存在しない場合は何も行わない。日時列に重複があり、単一の列で
    グループ化して重複を解消できない場合（例: 複数キーの組み合わせが必要なパネルデータ）も、
    誤ったコレログラムを出さないよう何も行わない。

    Args:
        df: 対象のDataFrame。
        dataset_name: 出力ファイル名・図のタイトルに使うデータセット識別名。
        figures_dir: 図の出力先ディレクトリ。
        moving_average_window: 移動平均の窓幅（点数）。
        seasonal_period: 季節差分・季節対数差分の周期（ラグ数）。
        nlags: ACF/PACFで計算・表示するラグ数の上限。
        max_plot_points: 生値と移動平均の図で1系列あたりの最大プロット点数。
        max_acf_points: ACF/PACF計算に用いる最大点数。

    Returns:
        `"{グループ名}:{カラム名}"` の形式で、図を作成した (グループ, カラム) の一覧。
        対象外だった場合は空リスト。
    """
    dt_col = find_datetime_column(df)
    if dt_col is None:
        return []

    group_col = find_grouping_column(df, dt_col)
    if group_col is None and df[dt_col].n_unique() < df.height:
        # 日時が重複しているが単一列でグループ化できないため対象外とする
        return []

    if group_col is None:
        groups = {(dataset_name,): df}
    else:
        partitions = df.partition_by(group_col, as_dict=True)
        groups = {(f"{dataset_name}_{key[0]}",): sub_df for key, sub_df in partitions.items()}

    processed: list[str] = []
    for (group_name,), group_df in groups.items():
        sorted_df = group_df.sort(dt_col)
        dt = sorted_df[dt_col]
        numeric_cols = find_numeric_columns(sorted_df, exclude=(dt_col,))

        for col in numeric_cols:
            values = sorted_df[col]
            series_dict = build_transformed_series(values, seasonal_period)
            base = f"{group_name}__{col}"

            plot_series_with_moving_average(
                dt,
                series_dict,
                group_name,
                col,
                figures_dir / f"{base}__series_with_moving_average.png",
                moving_average_window=moving_average_window,
                max_points=max_plot_points,
            )
            plot_acf_correlogram(
                series_dict,
                group_name,
                col,
                figures_dir / f"{base}__acf_correlogram.png",
                nlags=nlags,
                max_points=max_acf_points,
            )
            plot_pacf_correlogram(
                series_dict,
                group_name,
                col,
                figures_dir / f"{base}__pacf_correlogram.png",
                nlags=nlags,
                max_points=max_acf_points,
            )
            processed.append(f"{group_name}:{col}")

    return processed
