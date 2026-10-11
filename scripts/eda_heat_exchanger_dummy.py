"""`data/raw/dummy/`（熱交換器10基の運転データ・ダミー）の探索的データ分析（EDA）を行うスクリプト。

`src/` 配下のモジュールを使い、汎用のデータ品質確認・時系列診断に加えて、このデータ固有の
確認（運転・停止の状況、欠測・外れ値、原油比率の整合、汚れ係数の推移と増え方の要因、
学習・テスト期間の違い）を行う。

入力（`data/raw/dummy/`。読み取りのみで書き込まない）:
    - `train.csv`: 学習期間（2019〜2024年）の日次データ。横持ち（`日付` と `HX01_汚れ係数` など）
    - `test.csv`: テスト期間（2025年）。目的変数 `HXnn_汚れ係数` は null
    - `events.csv`: 停止・洗浄の記録
    - `crude.csv`: 原油A〜Cの性状
    - `chemical.csv`: 薬剤タイプA・タイプBの性状
    - `test_answer.csv` は正解なので、EDA では行数の確認以外に使わない（中身を見るとリークになる）

処理の流れ:
    1. 読み込み（`util.csv_io.read_csv_auto`）と、ファイルごとの概要・日付の連続性
    2. 横持ち → 縦持ち（1行 = 1日 × 1機器）への変換と、派生列 `圧力損失`（入口 − 出口）の追加
    3. データ品質確認（`eda.csv_quality.run_quality_checks`。縦持ちの学習データ）
    4. 時系列診断（`eda.time_series_eda.run_time_series_checks`。機器ごとの汚れ係数・圧力損失）
    5. このデータ固有の確認（本スクリプト内の関数）
       - 運転・停止の状況と欠測率（機器 × 列）、停止日に測定値が残っていないか
       - 原油比率の整合（A + B + C = 1.0、欠測が3列そろっているか）
       - 外れ値の候補（前後の日の中央値から大きく離れた値）
       - 汚れ係数の推移（洗浄の記録と重ねる）・圧力損失との関係
       - 汚れ係数の増え方（14日間の平均）と、原油の性状・薬剤・流量との関係
       - 停止・洗浄の記録の要約
       - 学習期間とテスト期間の特徴量の分布の違い

Usage:
    uv run python scripts/eda_heat_exchanger_dummy.py
    uv run python scripts/eda_heat_exchanger_dummy.py --output-root outputs/eda_dummy

出力（既定のルートは `outputs/`。ファイル名はすべて `dummy_hx__` で始まる）:
    - `tables/`: 概要・欠測率・外れ値の候補・汚れ係数の要約・増え方の集計・期間の比較など
    - `figures/`: 品質確認の図・時系列診断の図・汚れ係数の推移・増え方の要因・期間の比較
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# scripts/ から直接実行しても src/ 配下のパッケージをimportできるようにする
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

# 図はファイル保存のみ行うため非対話型バックエンドに固定する
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from eda.csv_quality import run_quality_checks
from eda.time_series_eda import run_time_series_checks
from util.csv_io import read_csv_auto
from util.paths import data_dir, ensure_parent_dir, outputs_dir
from util.plotting import add_caption, ensure_japanese_font

# --- 定数 --------------------------------------------------------------------------

# 入力ディレクトリ（data/raw は読み取り専用として扱い、書き込まない）
RAW_DIR = data_dir() / "raw" / "dummy"
DATASET_NAME = "dummy_hx"  # 出力ファイル名の先頭に付ける識別名
TIME_COL = "日付"
UNIT_COL = "機器"
TARGET = "汚れ係数"
EXCHANGERS = [f"HX{i:02d}" for i in range(1, 11)]

# 機器ごとの列（横持ちでは `HXnn_<列名>`）
MEASUREMENTS = (
    "汚れ係数",
    "入口圧力",
    "出口圧力",
    "高温側流量",
    "低温側流量",
    "原油A比率",
    "原油B比率",
    "原油C比率",
    "薬剤種類",
)
RATIO_COLUMNS = ("原油A比率", "原油B比率", "原油C比率")
SENSOR_COLUMNS = ("入口圧力", "出口圧力", "高温側流量", "低温側流量")

# 図に使う単位
UNITS = {
    "汚れ係数": "m²·K/kW",
    "入口圧力": "MPa",
    "出口圧力": "MPa",
    "圧力損失": "MPa",
    "高温側流量": "t/h",
    "低温側流量": "t/h",
}

# 時系列診断のパラメータ（日次データ向け）
MOVING_AVERAGE_WINDOW = 30  # 約1か月の移動平均
SEASONAL_PERIOD = 365  # 年周期の季節差分
NLAGS = 60  # ACF/PACFで表示するラグ数（約2か月）

# 外れ値の候補: 前後 OUTLIER_WINDOW 日の中央値から、相対的に OUTLIER_THRESHOLD 以上離れた値
OUTLIER_WINDOW = 15
OUTLIER_THRESHOLD = 0.25
# 汚れ係数の増え方を測る期間（日）。1日ごとの変化は測定誤差に埋もれるため、期間の平均で見る
GROWTH_HORIZON = 14
# 期間中の変化がこれより小さい（大きく下がった）期間は洗浄をはさむとみなし、増え方の集計から除く
CLEANING_DROP = -0.1

# 停止日（低温側流量が 0）。流量が欠測の日はセンサーの欠測とみなし、運転中として扱う
STOPPED = (pl.col("低温側流量") == 0).fill_null(False)


# --- 読み込みと変換 --------------------------------------------------------------------


def load_dummy(raw_dir: Path = RAW_DIR) -> dict[str, pl.DataFrame]:
    """`data/raw/dummy/` の CSV を読み込む（`test_answer.csv` は行数の確認のためだけに読む）。

    Args:
        raw_dir: 入力ディレクトリ。

    Returns:
        ファイル名（拡張子なし）→ DataFrame。日付の列は Datetime 型になる。
    """
    names = ("train", "test", "events", "crude", "chemical", "test_answer")
    frames = {name: read_csv_auto(raw_dir / f"{name}.csv") for name in names}
    for name in ("train", "test", "test_answer"):
        frames[name] = frames[name].sort(TIME_COL)
    return frames


def to_long(wide: pl.DataFrame) -> pl.DataFrame:
    """横持ち（`日付`, `HX01_汚れ係数`, …）を縦持ち（1行 = 1日 × 1機器）にする。

    派生列 `圧力損失`（入口圧力 − 出口圧力）を加える。汚れると流路が狭まり大きくなるため、
    目的変数の手がかりになる。

    Args:
        wide: 横持ちの DataFrame。

    Returns:
        列 `日付`, `機器`, 各測定値, `圧力損失` の DataFrame（日付・機器の順）。
    """
    parts = []
    for unit in EXCHANGERS:
        parts.append(
            wide.select(
                pl.col(TIME_COL),
                pl.lit(unit).alias(UNIT_COL),
                *[pl.col(f"{unit}_{m}").alias(m) for m in MEASUREMENTS],
            ).with_columns(
                # テストの目的変数は全行が空で文字列と判定されるため、数値の列は Float64 にそろえる
                *[pl.col(m).cast(pl.Float64) for m in MEASUREMENTS if m != "薬剤種類"],
                pl.col("薬剤種類").cast(pl.String),
            )
        )
    return (
        pl.concat(parts)
        .with_columns((pl.col("入口圧力") - pl.col("出口圧力")).alias("圧力損失"))
        .sort(TIME_COL, UNIT_COL)
    )


# --- このデータ固有の確認 -------------------------------------------------------------


def file_overview(frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """ファイルごとの行数・列数と、日付の列があればその期間を一覧にする。"""
    rows = []
    for name, df in frames.items():
        start = end = None
        if TIME_COL in df.columns:
            start, end = df[TIME_COL].min(), df[TIME_COL].max()
        rows.append(
            {
                "file": f"{name}.csv",
                "n_rows": df.height,
                "n_columns": df.width,
                "start": start,
                "end": end,
            }
        )
    return pl.DataFrame(rows)


def date_continuity_summary(df: pl.DataFrame, name: str) -> dict[str, object]:
    """日付が1日刻みで欠けも重複もなく並んでいるかを要約する（時系列モデルの前提の確認）。"""
    dates = df[TIME_COL].cast(pl.Date)
    expected = pl.date_range(dates.min(), dates.max(), interval="1d", eager=True)
    return {
        "file": name,
        "start": dates.min(),
        "end": dates.max(),
        "n_rows": df.height,
        "expected_days": expected.len(),
        "missing_days": expected.filter(~expected.is_in(dates.implode())).len(),
        "duplicated_days": int(dates.is_duplicated().sum()),
    }


def operation_summary(long: pl.DataFrame) -> pl.DataFrame:
    """機器ごとの運転・停止日数、運転中の欠測率、停止日に測定値が残っている件数を集計する。

    停止日は `低温側流量 == 0` の日。停止中は流量以外の測定値が欠測になるはずなので、
    値が残っていればデータの不整合を疑う。

    Returns:
        1行 = 1機器。列 `stop_days`, `running_days`, `missing_<列>`（運転中の欠測率 %）,
        `stop_with_values`（停止日に流量以外の値がある件数）。
    """
    stopped = STOPPED
    checked = (TARGET, "入口圧力", "出口圧力", *RATIO_COLUMNS)
    return (
        long.group_by(UNIT_COL)
        .agg(
            stopped.sum().alias("stop_days"),
            (~stopped).sum().alias("running_days"),
            *[
                (pl.col(c).filter(~stopped).is_null().mean() * 100).round(2).alias(f"missing_{c}")
                for c in (TARGET, *SENSOR_COLUMNS, RATIO_COLUMNS[0])
            ],
            pl.sum_horizontal(
                [pl.col(c).filter(stopped).is_not_null().sum() for c in checked]
            ).alias("stop_with_values"),
        )
        .sort(UNIT_COL)
    )


def ratio_check(long: pl.DataFrame) -> pl.DataFrame:
    """原油比率の整合を確認する（A + B + C = 1.0、範囲 0〜1、欠測が3列そろっているか）。"""
    a, b, c = (pl.col(name) for name in RATIO_COLUMNS)
    present = a.is_not_null() & b.is_not_null() & c.is_not_null()
    partial = (a.is_null() | b.is_null() | c.is_null()) & ~(a.is_null() & b.is_null() & c.is_null())
    return long.select(
        present.sum().alias("n_rows_with_ratios"),
        partial.sum().alias("n_rows_partially_missing"),
        (a + b + c - 1.0).abs().filter(present).max().alias("max_abs_sum_error"),
        (pl.min_horizontal(a, b, c).filter(present) < 0).sum().alias("n_negative"),
        (pl.max_horizontal(a, b, c).filter(present) > 1).sum().alias("n_above_one"),
    )


def outlier_candidates(
    long: pl.DataFrame,
    columns: tuple[str, ...] = SENSOR_COLUMNS,
    window: int = OUTLIER_WINDOW,
    threshold: float = OUTLIER_THRESHOLD,
) -> pl.DataFrame:
    """前後の日の中央値から大きく離れた値を、外れ値の候補として返す（運転中の日のみ）。

    機器ごとに、前後 `window` 日の移動中央値との相対差 |値 / 中央値 − 1| が `threshold` 以上の値。
    急な運転変更でも起こりうるため、あくまで候補（確認用）とする。

    Returns:
        列 `日付`, `機器`, `column`, `value`, `rolling_median`, `relative_gap` の表。
    """
    running = long.filter(~STOPPED)
    frames = []
    for column in columns:
        median = (
            pl.col(column)
            .rolling_median(window, center=True, min_samples=window // 2)
            .over(UNIT_COL, order_by=TIME_COL)
        )
        frames.append(
            running.select(
                TIME_COL, UNIT_COL, pl.col(column).alias("value"), median.alias("rolling_median")
            )
            .with_columns(
                pl.lit(column).alias("column"),
                (pl.col("value") / pl.col("rolling_median") - 1).abs().alias("relative_gap"),
            )
            .filter(pl.col("relative_gap") >= threshold)
        )
    columns_order = [TIME_COL, UNIT_COL, "column", "value", "rolling_median", "relative_gap"]
    return pl.concat(frames).select(columns_order).sort(TIME_COL, UNIT_COL, "column")


def target_summary(long: pl.DataFrame) -> pl.DataFrame:
    """機器ごとの汚れ係数の要約（平均・中央値・最大・欠測日数）と、圧力損失との順位相関。"""
    return (
        long.group_by(UNIT_COL)
        .agg(
            pl.col(TARGET).count().alias("n_observed"),
            pl.col(TARGET).mean().round(4).alias("mean"),
            pl.col(TARGET).median().round(4).alias("median"),
            pl.col(TARGET).max().round(4).alias("max"),
            pl.corr(pl.col(TARGET), pl.col("圧力損失"), method="spearman")
            .round(3)
            .alias("spearman_vs_pressure_drop"),
        )
        .sort(UNIT_COL)
    )


def weighted_crude_properties(long: pl.DataFrame, crude: pl.DataFrame) -> pl.DataFrame:
    """原油比率で重み付けした、その日の原油の性状（加重アスファルテン分・硫黄分・API比重）を加える。

    例: 加重アスファルテン分 = Σ_k 原油k比率 × 原油kのアスファルテン分。
    """
    props = {row["原油"]: row for row in crude.iter_rows(named=True)}
    exprs = []
    for prop, alias in (
        ("アスファルテン分", "加重アスファルテン分"),
        ("全硫黄分", "加重硫黄分"),
        ("API比重", "加重API比重"),
    ):
        exprs.append(
            sum(pl.col(f"原油{k}比率") * props[k][prop] for k in ("A", "B", "C")).alias(alias)  # type: ignore[arg-type]
        )
    return long.with_columns(exprs)


def fouling_growth(
    long: pl.DataFrame, crude: pl.DataFrame, horizon: int = GROWTH_HORIZON
) -> pl.DataFrame:
    """汚れ係数の増え方（`horizon` 日間の平均、1日あたり）と、その期間の運転条件を並べる。

    1日ごとの変化（約0.004）は汚れ係数の測定誤差（約0.02）より小さく、そのままでは要因との
    関係が見えない。そこで `(horizon 日後の値 − 当日の値) / horizon` を増え方とし、
    原油の性状・流量はその期間の平均にする（薬剤は期間の初日の種類）。洗浄をはさむ期間
    （増え方が `CLEANING_DROP / horizon` 未満）は除く。縦持ちは機器ごとに全日付の行が
    そろっているため、行のずれ＝日数のずれになる。

    Returns:
        列 `日付`（期間の初日）, `機器`, `growth`, `薬剤種類`, 期間平均の `原油C比率` ・
        `低温側流量` ・加重性状 の表。
    """
    drivers = ("原油C比率", "低温側流量", "加重アスファルテン分", "加重硫黄分", "加重API比重")
    data = weighted_crude_properties(long, crude).sort(UNIT_COL, TIME_COL)
    future = pl.col(TARGET).shift(-horizon).over(UNIT_COL)
    # 当日〜(horizon − 1) 日後の平均（前向きの移動平均。欠測・停止が半分を超える期間は null）
    window_mean = [
        pl.col(d)
        .rolling_mean(horizon, min_samples=horizon // 2)
        .shift(-(horizon - 1))
        .over(UNIT_COL)
        .alias(d)
        for d in drivers
    ]
    return (
        data.with_columns(((future - pl.col(TARGET)) / horizon).alias("growth"), *window_mean)
        .filter(pl.col("growth").is_not_null() & (pl.col("growth") > CLEANING_DROP / horizon))
        .select(TIME_COL, UNIT_COL, "growth", "薬剤種類", *drivers)
        .drop_nulls()
    )


def growth_by_chemical(growth: pl.DataFrame, chemical: pl.DataFrame) -> pl.DataFrame:
    """薬剤ごとの汚れ係数の増え方（1日あたり、機器を通した平均）と、薬剤の性状を並べる。"""
    return (
        growth.group_by("薬剤種類")
        .agg(
            pl.len().alias("n_days"),
            (pl.col("growth").mean() * 1000).round(3).alias("mean_growth_e-3_per_day"),
            (pl.col("growth").median() * 1000).round(3).alias("median_growth_e-3_per_day"),
        )
        .join(chemical, on="薬剤種類", how="left")
        .sort("薬剤種類")
    )


def growth_correlations(growth: pl.DataFrame) -> pl.DataFrame:
    """増え方と運転条件の順位相関（機器ごと・全体）。

    機器によって温度（汚れやすさ）が違うため、機器ごとの相関も見る。
    """
    drivers = ("原油C比率", "加重アスファルテン分", "加重硫黄分", "加重API比重", "低温側流量")
    overall = growth.select(
        pl.lit("全体").alias(UNIT_COL),
        *[pl.corr("growth", d, method="spearman").round(3).alias(d) for d in drivers],
    )
    per_unit = (
        growth.group_by(UNIT_COL)
        .agg(*[pl.corr("growth", d, method="spearman").round(3).alias(d) for d in drivers])
        .sort(UNIT_COL)
    )
    return pl.concat([overall, per_unit])


def events_summary(events: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """停止・洗浄の記録を要約する。

    Returns:
        (機器 × 種類 の件数表, 洗浄の間隔（日）と洗浄前の汚れ係数の要約（機器ごと）)。
    """
    counts = (
        events.pivot(on="種類", index="機器", values="開始日", aggregate_function="len")
        .fill_null(0)
        .sort("機器")
    )
    cleanings = events.filter(pl.col("種類") != "突発停止").sort("機器", "開始日")
    intervals = (
        cleanings.with_columns(
            (pl.col("開始日") - pl.col("終了日").shift(1).over("機器"))
            .dt.total_days()
            .alias("interval_days")
        )
        .group_by("機器")
        .agg(
            pl.len().alias("n_cleanings"),
            pl.col("interval_days").mean().round(1).alias("mean_interval_days"),
            pl.col("停止前の汚れ係数").mean().round(3).alias("mean_fouling_before_cleaning"),
            pl.col("停止前の汚れ係数").min().round(3).alias("min_fouling_before_cleaning"),
        )
        .sort("機器")
    )
    return counts, intervals


def train_test_shift(train_long: pl.DataFrame, test_long: pl.DataFrame) -> pl.DataFrame:
    """運転中の日の特徴量の平均・標準偏差を、学習期間とテスト期間で比べる。

    差（標準化平均差 = 平均の差 / 学習期間の標準偏差）が大きい列は、テスト期間で
    学習期間と違う運転条件になっている可能性がある。
    """
    columns = ("入口圧力", "出口圧力", "圧力損失", "高温側流量", "低温側流量", *RATIO_COLUMNS)
    rows = []
    for col in columns:
        tr = train_long.filter(~STOPPED)[col].drop_nulls()
        te = test_long.filter(~STOPPED)[col].drop_nulls()
        std = tr.std()
        rows.append(
            {
                "column": col,
                "train_mean": round(tr.mean(), 4),
                "test_mean": round(te.mean(), 4),
                "train_std": round(std, 4),
                "test_std": round(te.std(), 4),
                "standardized_mean_diff": round((te.mean() - tr.mean()) / std, 3) if std else None,
            }
        )
    return pl.DataFrame(rows)


def chemical_share(long: pl.DataFrame) -> pl.DataFrame:
    """機器 × 薬剤の、運転中の日数の割合（%）。"""
    running = long.filter(~STOPPED)
    return (
        running.group_by(UNIT_COL, "薬剤種類")
        .len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over(UNIT_COL) * 100).round(1).alias("share_%")
        )
        .pivot(on="薬剤種類", index=UNIT_COL, values="share_%")
        .fill_null(0.0)
        .sort(UNIT_COL)
    )


# --- 図 ------------------------------------------------------------------------------


def _setup_style() -> None:
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)
    # sns.set_theme がフォント設定を上書きするため、その後に日本語フォントを設定する
    ensure_japanese_font()


def plot_fouling_timeseries(long: pl.DataFrame, events: pl.DataFrame) -> plt.Figure:
    """機器ごとの汚れ係数の推移（10段）。洗浄・定期修理・突発停止の期間を色で重ねる。"""
    _setup_style()
    fig, axes = plt.subplots(
        len(EXCHANGERS),
        1,
        figsize=(14, 2.0 * len(EXCHANGERS)),
        sharex=True,
        constrained_layout=True,
    )
    colors = {"洗浄停止": "#c44e52", "定期修理（洗浄）": "#8172b3", "突発停止": "#bbbbbb"}
    for ax, unit in zip(axes, EXCHANGERS, strict=True):
        series = long.filter(pl.col(UNIT_COL) == unit)
        ax.plot(series[TIME_COL], series[TARGET], lw=0.8, color="#4c72b0")
        for row in events.filter(pl.col("機器") == unit).iter_rows(named=True):
            ax.axvspan(
                row["開始日"],
                row["終了日"],
                color=colors.get(row["種類"], "#cccccc"),
                alpha=0.5,
                lw=0,
            )
        ax.set_ylabel(unit)
        ax.set_ylim(bottom=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in colors.values()]
    axes[0].legend(handles, list(colors), loc="upper left", ncols=3, fontsize=8)
    fig.suptitle(f"汚れ係数（{UNITS[TARGET]}）の推移と停止・洗浄の期間（学習期間）")
    add_caption(
        fig,
        f"{long[TIME_COL].min():%Y-%m-%d}〜{long[TIME_COL].max():%Y-%m-%d}。"
        "色の帯は events.csv の停止期間",
    )
    return fig


def plot_fouling_vs_pressure_drop(long: pl.DataFrame) -> plt.Figure:
    """機器ごとの「汚れ係数 vs 圧力損失」の散布図（2行5列）。外れ値に強い順位相関を表示する。"""
    _setup_style()
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True, constrained_layout=True)
    for ax, unit in zip(axes.flat, EXCHANGERS, strict=True):
        data = long.filter(pl.col(UNIT_COL) == unit).select(TARGET, "圧力損失").drop_nulls()
        # 圧力センサーの外れ値は縦軸を大きく引き伸ばすため、描画は 1〜99% 点の範囲に絞る
        low, high = data["圧力損失"].quantile(0.01), data["圧力損失"].quantile(0.99)
        shown = data.filter(pl.col("圧力損失").is_between(low, high))
        ax.scatter(shown[TARGET], shown["圧力損失"], s=4, alpha=0.3, color="#4c72b0")
        rho = data.select(pl.corr(TARGET, "圧力損失", method="spearman")).item()
        ax.set_title(f"{unit}（順位相関 {rho:.2f}）", fontsize=10)
        ax.set_xlabel(f"汚れ係数 ({UNITS[TARGET]})")
        ax.set_ylabel(f"圧力損失 ({UNITS['圧力損失']})")
    fig.suptitle("汚れ係数と圧力損失（入口圧力 − 出口圧力）の関係（学習期間、運転中の日）")
    add_caption(
        fig, "圧力損失の 1〜99% 点の外（センサーの外れ値を含む）は描画から除外。相関は全点で計算"
    )
    return fig


def plot_growth_drivers(growth: pl.DataFrame) -> plt.Figure:
    """汚れ係数の増え方と、薬剤・原油の性状・流量の関係（3枚組）。"""
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    scaled = growth.with_columns((pl.col("growth") * 1000).alias("growth_1000"))
    # 1. 薬剤ごとの増え方（箱ひげ図。1日ごとの値は測定誤差でばらつくので中央値・四分位で見る）
    order = ["なし", "タイプA", "タイプB"]
    sns.boxplot(
        data=scaled.select("薬剤種類", "growth_1000").to_pandas(),
        x="薬剤種類",
        y="growth_1000",
        order=order,
        ax=axes[0],
        color="#8fb3d9",
        showfliers=False,
    )
    axes[0].axhline(0, color="gray", lw=0.8)
    axes[0].set_title("薬剤ごとの増え方")
    axes[0].set_ylabel("汚れ係数の増え方（×10⁻³ / 日）")
    # 2-3. 加重アスファルテン分・低温側流量を10区間に分けた平均（測定誤差を平均で打ち消す）
    for ax, driver, label in (
        (axes[1], "加重アスファルテン分", "加重アスファルテン分（質量%）"),
        (axes[2], "低温側流量", "低温側流量（t/h）"),
    ):
        binned = (
            scaled.drop_nulls(driver)
            .with_columns(pl.col(driver).qcut(10, labels=[str(i) for i in range(10)]).alias("bin"))
            .group_by("bin")
            .agg(
                pl.col(driver).mean().alias("x"),
                pl.col("growth_1000").mean().alias("y"),
                pl.len().alias("n"),
            )
            .sort("x")
        )
        ax.plot(binned["x"], binned["y"], marker="o", color="#c44e52")
        ax.set_title(f"{driver}（10分位ごとの平均）")
        ax.set_xlabel(label)
        ax.set_ylabel("汚れ係数の増え方の平均（×10⁻³ / 日）")
    fig.suptitle(f"汚れ係数の増え方（{GROWTH_HORIZON}日間の平均）と運転条件（学習期間、全機器）")
    add_caption(
        fig,
        f"n={growth.height:,}。増え方は {GROWTH_HORIZON} 日間の平均（洗浄をはさむ期間は除く）。"
        "運転条件はその期間の平均",
    )
    return fig


def plot_train_test_distributions(train_long: pl.DataFrame, test_long: pl.DataFrame) -> plt.Figure:
    """学習期間とテスト期間の特徴量の分布（運転中の日、全機器）を重ねて比べる。"""
    _setup_style()
    columns = ("圧力損失", "高温側流量", "低温側流量", "原油A比率", "原油B比率", "原油C比率")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    for ax, col in zip(axes.flat, columns, strict=True):
        tr = train_long.filter(~STOPPED)[col].drop_nulls()
        te = test_long.filter(~STOPPED)[col].drop_nulls()
        low, high = tr.quantile(0.005), tr.quantile(0.995)  # 外れ値で横軸が伸びないようにする
        bins = np.linspace(low, high, 50)
        ax.hist(
            tr.filter(tr.is_between(low, high)),
            bins=bins,
            density=True,
            alpha=0.5,
            label="学習（2019〜2024）",
        )
        ax.hist(
            te.filter(te.is_between(low, high)),
            bins=bins,
            density=True,
            alpha=0.5,
            label="テスト（2025）",
        )
        ax.set_title(col)
        ax.set_xlabel(f"{col} ({UNITS[col]})" if col in UNITS else col)
        ax.set_ylabel("密度")
    axes.flat[0].legend()
    fig.suptitle("学習期間とテスト期間の特徴量の分布（運転中の日、全機器）")
    add_caption(fig, "学習期間の 0.5〜99.5% 点の範囲で描画（外れ値を除く）")
    return fig


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """図を保存して閉じる。"""
    fig.savefig(ensure_parent_dir(path), dpi=150, bbox_inches="tight")
    plt.close(fig)  # メモリを解放する
    return path


# --- メイン処理 ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """EDA一式を実行し、表と図を保存する。

    Args:
        argv: コマンドライン引数（Noneなら `sys.argv` を使う。テストから呼ぶ場合に指定）。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR, help="入力ディレクトリ（既定: data/raw/dummy）"
    )
    parser.add_argument(
        "--output-root", type=Path, default=None, help="出力のルート（既定: outputs/）"
    )
    args = parser.parse_args(argv)
    root = args.output_root or outputs_dir()
    figures_dir, tables_dir = root / "figures", root / "tables"

    def save_table(frame: pl.DataFrame, name: str) -> None:
        frame.write_csv(ensure_parent_dir(tables_dir / f"{DATASET_NAME}__{name}.csv"))

    # 1. 読み込みと、ファイルごとの概要・日付の連続性
    frames = load_dummy(args.raw_dir)
    overview = file_overview(frames)
    save_table(overview, "file_overview")
    print("=== ファイルの概要 ===")
    print(overview)
    continuity = pl.DataFrame([date_continuity_summary(frames[n], n) for n in ("train", "test")])
    save_table(continuity, "date_continuity")
    print("=== 日付の連続性 ===")
    print(continuity)

    # 2. 縦持ちに変換（1行 = 1日 × 1機器）
    train_long = to_long(frames["train"])
    test_long = to_long(frames["test"])
    print(f"=== 縦持ち: 学習 {train_long.height:,}行、テスト {test_long.height:,}行 ===")
    print(f"テストの目的変数がすべて null: {test_long[TARGET].null_count() == test_long.height}")

    # 3. データ品質確認（縦持ちの学習データ）
    df_overview, col_overview = run_quality_checks(
        train_long,
        f"{DATASET_NAME}_train_long",
        source_path=str(args.raw_dir / "train.csv"),
        figures_dir=figures_dir,
        tables_dir=tables_dir,
    )
    print("=== 全体概要（縦持ちの学習データ） ===")
    print(df_overview)
    print(col_overview.select("column", "dtype", "missing_rate", "n_unique", "mean", "min", "max"))

    # 4. 時系列診断（機器ごとの汚れ係数・圧力損失。機器列でグループに分けて計算される）
    processed = run_time_series_checks(
        train_long.select(TIME_COL, UNIT_COL, TARGET, "圧力損失"),
        DATASET_NAME,
        figures_dir,
        moving_average_window=MOVING_AVERAGE_WINDOW,
        seasonal_period=SEASONAL_PERIOD,
        nlags=NLAGS,
    )
    print(f"=== 時系列診断: {len(processed)}系列 ===")

    # 5. このデータ固有の確認
    operation = operation_summary(train_long)
    save_table(operation, "operation_summary")
    print("=== 運転・停止と欠測率（%、運転中の日） ===")
    print(operation)

    ratios = ratio_check(pl.concat([train_long, test_long]))
    save_table(ratios, "ratio_check")
    print("=== 原油比率の整合 ===")
    print(ratios)

    outliers = outlier_candidates(train_long)
    save_table(outliers, "outlier_candidates")
    print(f"=== 外れ値の候補: {outliers.height}件 ===")
    print(outliers.group_by("column").len().sort("column"))

    target = target_summary(train_long)
    save_table(target, "target_summary")
    print("=== 汚れ係数の要約と圧力損失との順位相関 ===")
    print(target)

    events = frames["events"]
    counts, intervals = events_summary(events)
    save_table(counts, "event_counts")
    save_table(intervals, "cleaning_intervals")
    print("=== 停止・洗浄の件数 ===")
    print(counts)
    print("=== 洗浄の間隔と洗浄前の汚れ係数 ===")
    print(intervals)

    share = chemical_share(train_long)
    save_table(share, "chemical_share")
    growth = fouling_growth(train_long, frames["crude"])
    by_chemical = growth_by_chemical(growth, frames["chemical"])
    correlations = growth_correlations(growth)
    save_table(by_chemical, "growth_by_chemical")
    save_table(correlations, "growth_correlations")
    print("=== 薬剤ごとの汚れ係数の増え方（×10⁻³ / 日） ===")
    print(by_chemical)
    print("=== 増え方と運転条件の順位相関 ===")
    print(correlations)

    shift = train_test_shift(train_long, test_long)
    save_table(shift, "train_test_shift")
    print("=== 学習期間とテスト期間の比較（運転中の日） ===")
    print(shift)

    save_figure(
        plot_fouling_timeseries(train_long, events),
        figures_dir / f"{DATASET_NAME}__fouling_timeseries.png",
    )
    save_figure(
        plot_fouling_vs_pressure_drop(train_long),
        figures_dir / f"{DATASET_NAME}__fouling_vs_pressure_drop.png",
    )
    save_figure(plot_growth_drivers(growth), figures_dir / f"{DATASET_NAME}__growth_drivers.png")
    save_figure(
        plot_train_test_distributions(train_long, test_long),
        figures_dir / f"{DATASET_NAME}__train_test_distributions.png",
    )

    n_figures = len(list(figures_dir.glob(f"{DATASET_NAME}*.png")))
    n_tables = len(list(tables_dir.glob(f"{DATASET_NAME}*.csv")))
    print(f"=== 出力: 図 {n_figures}枚（{figures_dir}）, 表 {n_tables}件（{tables_dir}） ===")


if __name__ == "__main__":
    main()
