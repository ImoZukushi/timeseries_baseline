"""`data/raw/DailyDelhiClimateTrain.csv` の探索的データ分析（EDA）を行うスクリプト。

`src/` 配下のモジュールを使い、汎用のデータ品質確認・時系列診断に加えて、
このデータ固有の確認（日付の連続性・物理的にありうる範囲・月別の季節性・年別の推移）を行う。

処理の流れ:
    1. 読み込み（`util.csv_io.read_csv_auto`: エンコーディング自動判定・日付列の自動パース）
    2. データ品質確認（`eda.csv_quality.run_quality_checks`）
       - データフレーム全体の概要（行数・列数・重複・欠損）とカラム単位の要約統計
       - 欠損の全体像・ヒストグラム・相関ヒートマップ・記録パターン・散布図行列
    3. 時系列診断（`eda.time_series_eda.run_time_series_checks`）
       - 変数ごとに、原系列・対数・差分・対数差分・季節差分・季節対数差分の6系列について
         「生値と移動平均」「ACF」「PACF」の図を作成
       - 日次データのため、移動平均は30日、季節差分の周期は365日（年周期）とする
    4. このデータ固有の確認（本スクリプト内の関数）
       - 日付の連続性（欠けている日・重複している日の有無）
       - 物理的にありうる範囲の外にある値（測定・記録の誤りの候補）
       - 月別の分布（箱ひげ図）で季節性を確認
       - 年別の平均で長期的な推移を確認

物理的にありうる範囲（本スクリプトの仮定。デリーの気候として明らかにおかしい値を検出する目安）:
    - meantemp（日平均気温, ℃）: -10〜50
    - humidity（湿度, %）: 0〜100
    - wind_speed（風速, km/h）: 0〜100
    - meanpressure（平均気圧, hPa）: 950〜1050

Usage:
    uv run python scripts/eda_delhi_climate.py
    uv run python scripts/eda_delhi_climate.py --output-root outputs/eda_delhi

出力（既定のルートは `outputs/`。ファイル名はすべて `delhi_climate_train__` で始まる）:
    - `tables/`: 全体概要・カラム要約・日付の連続性・範囲外の値・年別平均
    - `figures/`: 品質確認の図・時系列診断の図・月別の箱ひげ図
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# scripts/ から直接実行しても src/ 配下のパッケージをimportできるようにする
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

# 図はファイル保存のみ行うため非対話型バックエンドに固定する
# （matplotlib.pyplot を最初にimportする前に設定する必要がある）
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from eda.csv_quality import run_quality_checks
from eda.time_series_eda import run_time_series_checks
from util.csv_io import read_csv_auto
from util.paths import data_dir, ensure_parent_dir, outputs_dir
from util.plotting import add_caption, ensure_japanese_font

# --- 定数 --------------------------------------------------------------------------

# 入力ファイル（data/raw は読み取り専用として扱い、書き込まない）
TRAIN_PATH = data_dir() / "raw" / "DailyDelhiClimateTrain.csv"
DATASET_NAME = "delhi_climate_train"  # 出力ファイル名の先頭に付ける識別名
TIME_COL = "date"

# 分析対象の数値変数と、図に使う日本語名・単位
VARIABLE_LABELS = {
    "meantemp": "日平均気温 (℃)",
    "humidity": "湿度 (%)",
    "wind_speed": "風速 (km/h)",
    "meanpressure": "平均気圧 (hPa)",
}

# 物理的にありうる範囲（下限, 上限）。範囲外は測定・記録の誤りの候補とみなす
PLAUSIBLE_RANGES = {
    "meantemp": (-10.0, 50.0),
    "humidity": (0.0, 100.0),
    "wind_speed": (0.0, 100.0),
    "meanpressure": (950.0, 1050.0),
}

# 時系列診断のパラメータ（日次データ向け）
MOVING_AVERAGE_WINDOW = 30  # 約1か月の移動平均で季節の推移を見る
SEASONAL_PERIOD = 365  # 年周期の季節差分
NLAGS = 60  # ACF/PACFで表示するラグ数（約2か月）


# --- このデータ固有の確認 -------------------------------------------------------------


def date_continuity_summary(df: pl.DataFrame, time_col: str = TIME_COL) -> pl.DataFrame:
    """日付が1日刻みで欠けも重複もなく並んでいるかを要約する。

    時系列モデル（ラグ特徴量など）は「1行 = 1日」を前提にするため、欠けている日や
    重複している日があると、ラグが意図した日数分ずれてしまう。

    Args:
        df: 日付列を持つDataFrame。
        time_col: 日付列の名前。

    Returns:
        1行の要約表（開始日・終了日・行数・期待される日数・欠けている日数・重複している日数）。
    """
    dates = df[time_col].cast(pl.Date)
    start, end = dates.min(), dates.max()
    # 開始日から終了日までの「あるべき日付」の一覧
    expected = pl.date_range(start, end, interval="1d", eager=True)
    missing = expected.filter(~expected.is_in(dates.implode()))
    return pl.DataFrame(
        {
            "start": [start],
            "end": [end],
            "n_rows": [df.height],
            "expected_days": [expected.len()],
            "missing_days": [missing.len()],
            "duplicated_days": [int(dates.is_duplicated().sum())],
        }
    )


def out_of_range_values(
    df: pl.DataFrame, ranges: dict[str, tuple[float, float]], time_col: str = TIME_COL
) -> pl.DataFrame:
    """物理的にありうる範囲の外にある値を、日付・変数・値の一覧で返す。

    Args:
        df: 対象のDataFrame。
        ranges: 変数名 → (下限, 上限)。
        time_col: 日付列の名前。

    Returns:
        列 `date`, `variable`, `value`, `lower`, `upper` の表（該当が無ければ0行）。
    """
    frames = []
    for column, (lower, upper) in ranges.items():
        frames.append(
            df.filter(~pl.col(column).is_between(lower, upper)).select(
                pl.col(time_col),
                pl.lit(column).alias("variable"),
                pl.col(column).cast(pl.Float64).alias("value"),
                pl.lit(lower).alias("lower"),
                pl.lit(upper).alias("upper"),
            )
        )
    return pl.concat(frames).sort(time_col, "variable")


def yearly_means(
    df: pl.DataFrame, ranges: dict[str, tuple[float, float]], time_col: str = TIME_COL
) -> pl.DataFrame:
    """年ごとの平均と日数を集計する（長期的な推移や、年による欠けの確認用）。

    範囲外の値（記録の誤りの候補）は、1件でも平均を大きく歪めるため除外して平均する
    （例: 気圧 7679 hPa が1日あるだけで、その年の平均が約18 hPa上がる）。

    Args:
        df: 対象のDataFrame。
        ranges: 変数名 → (下限, 上限)。この範囲内の値だけで平均する。
        time_col: 日付列の名前。

    Returns:
        列 `year`, `n_days`, 各変数の平均 の表。
    """
    return (
        df.group_by(pl.col(time_col).dt.year().alias("year"))
        .agg(
            pl.len().alias("n_days"),
            *[
                # 範囲外の値を欠損にしてから平均する（mean は欠損を無視する）
                pl.col(v).filter(pl.col(v).is_between(lower, upper)).mean()
                for v, (lower, upper) in ranges.items()
            ],
        )
        .sort("year")
    )


def plot_monthly_boxplots(
    df: pl.DataFrame,
    ranges: dict[str, tuple[float, float]],
    time_col: str = TIME_COL,
) -> plt.Figure:
    """変数ごとに、月別の分布を箱ひげ図で描く（季節性の確認）。

    範囲外の値（記録の誤りの候補）は縦軸を大きく引き伸ばして分布が見えなくなるため、
    描画から除外する（除外した件数は注記に表示する）。

    Args:
        df: 対象のDataFrame。
        ranges: 変数名 → (下限, 上限)。
        time_col: 日付列の名前。

    Returns:
        作成した図。
    """
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
    # sns.set_theme がフォント設定を上書きするため、その後に日本語フォントを設定する
    ensure_japanese_font()
    variables = list(ranges)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    excluded_notes = []
    for ax, variable in zip(axes.flat, variables, strict=True):
        lower, upper = ranges[variable]
        in_range = df.filter(pl.col(variable).is_between(lower, upper))
        excluded = df.height - in_range.height
        if excluded:
            excluded_notes.append(f"{variable}: {excluded}件")
        plot_df = in_range.select(
            pl.col(time_col).dt.month().alias("month"), pl.col(variable)
        ).to_pandas()  # seabornに渡すためpandasに変換する
        sns.boxplot(data=plot_df, x="month", y=variable, ax=ax, color="#8fb3d9", fliersize=2)
        ax.set_title(f"{variable}: 月別の分布")
        ax.set_xlabel("月")
        ax.set_ylabel(VARIABLE_LABELS[variable])
    fig.suptitle(
        f"デリーの気候（学習データ）: 月別の分布 "
        f"（{df[time_col].min():%Y-%m-%d}〜{df[time_col].max():%Y-%m-%d}, n={df.height:,}日）"
    )
    note = "範囲外の値は描画から除外: " + ", ".join(excluded_notes) if excluded_notes else ""
    add_caption(fig, f"箱は四分位範囲、線は中央値、点は外れ値（1.5×IQRの外）。{note}")
    return fig


# --- メイン処理 ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """EDA一式を実行し、表と図を保存する。

    Args:
        argv: コマンドライン引数（Noneなら `sys.argv` を使う。テストから呼ぶ場合に指定）。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root", type=Path, default=None, help="出力のルート（既定: outputs/）"
    )
    args = parser.parse_args(argv)
    root = args.output_root or outputs_dir()
    figures_dir = root / "figures"
    tables_dir = root / "tables"

    # 1. 読み込み（日付列は Datetime 型として読み込まれる）
    df = read_csv_auto(TRAIN_PATH).sort(TIME_COL)
    print(f"読み込み: {TRAIN_PATH.name} {df.height}行 × {df.width}列")
    print(df.head())

    # 2. データ品質確認（全体概要・カラム要約・品質確認の図）
    df_overview, col_overview = run_quality_checks(
        df,
        DATASET_NAME,
        source_path=str(TRAIN_PATH.relative_to(data_dir() / "raw")),
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )
    print("=== 全体概要 ===")
    print(df_overview)
    print("=== カラム要約 ===")
    print(col_overview)

    # 3. 時系列診断（変数ごとに 6系列 × 3種類 の図）
    processed = run_time_series_checks(
        df,
        DATASET_NAME,
        figures_dir,
        moving_average_window=MOVING_AVERAGE_WINDOW,
        seasonal_period=SEASONAL_PERIOD,
        nlags=NLAGS,
    )
    print(f"=== 時系列診断: {len(processed)}変数 ===")
    print("  " + ", ".join(processed))

    # 4. このデータ固有の確認
    continuity = date_continuity_summary(df)
    continuity.write_csv(ensure_parent_dir(tables_dir / f"{DATASET_NAME}__date_continuity.csv"))
    print("=== 日付の連続性 ===")
    print(continuity)

    outliers = out_of_range_values(df, PLAUSIBLE_RANGES)
    outliers.write_csv(tables_dir / f"{DATASET_NAME}__out_of_range_values.csv")
    print(f"=== 物理的にありうる範囲の外の値: {outliers.height}件 ===")
    print(outliers)

    yearly = yearly_means(df, PLAUSIBLE_RANGES)
    yearly.write_csv(tables_dir / f"{DATASET_NAME}__yearly_means.csv")
    # 2017年は1日（2017-01-01）しかないため、他の年と平均を比べることはできない
    print("=== 年別の平均（範囲外の値を除く） ===")
    print(yearly)

    fig = plot_monthly_boxplots(df, PLAUSIBLE_RANGES)
    figure_path = ensure_parent_dir(figures_dir / f"{DATASET_NAME}__monthly_boxplots.png")
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # メモリを解放する

    n_figures = len(list(figures_dir.glob(f"{DATASET_NAME}*.png")))
    n_tables = len(list(tables_dir.glob(f"{DATASET_NAME}*.csv")))
    print(f"=== 出力: 図 {n_figures}枚（{figures_dir}）, 表 {n_tables}件（{tables_dir}） ===")


if __name__ == "__main__":
    main()
