"""`data/raw/dummy/`（熱交換器10基の運転データ・ダミー）を前処理し、学習用・テスト用のデータセットを作るスクリプト。

EDA（`scripts/eda_heat_exchanger_dummy.py`）で分かった点を踏まえ、縦持ち（1行 = 1日 × 1機器）の
データセットを作る。学習期間とテスト期間は同じ手順で処理する（洗浄の推定も `events.csv` を使わず、
運転データだけから行う。テスト期間には停止・洗浄の記録がないため）。

入力（`data/raw/dummy/`。読み取りのみで書き込まない）:
    - `train.csv` / `test.csv`: 学習期間（2019〜2024年）・テスト期間（2025年）の日次データ（横持ち）
    - `crude.csv` / `chemical.csv`: 原油A〜C・薬剤タイプA/Bの性状
    - `events.csv`: 停止・洗浄の記録（学習期間のみ）。洗浄の推定の答え合わせだけに使う
    - `test_answer.csv` は使わない

処理の流れ:
    1. 読み込みと縦持ちへの変換（EDA スクリプトの `load_dummy` ・ `to_long` を使う）。
       学習期間とテスト期間をつなげて機器ごとの連続した系列にする（テスト期間の初日の
       「前回の洗浄からの日数」などは、学習期間の最後から引き継ぐ必要があるため）
    2. クリーニング
       - 外れ値: 運転中の圧力・流量のうち、前後 15 日の移動中央値から 25% 以上離れた値を欠測にする
       - 補間: 運転中の圧力・流量・原油比率の欠測のうち、連続 3 日以内のものを機器ごとに線形補間する
         （停止期間をまたいでは補間しない。補間・外れ値の除去をした値には印の列を付ける）
       - 停止日（低温側流量 0）は行を残し、`停止` 列で区別する（測定値は欠測のまま）
    3. 洗浄の推定: 停止期間ごとに、日数（4日以上）または停止前後の圧力損失の低下（20% 以上）から、
       洗浄をしたかどうかを判定する（突発停止は 1〜3 日で、圧力損失はほとんど変わらない）
    4. 特徴量
       - 圧力損失（入口 − 出口。クリーニング後の値から計算）と、その 7 日移動平均
       - 原油比率で重み付けした原油の性状（アスファルテン分・硫黄分・API比重）
       - 薬剤の性状（`chemical.csv` を結合。薬剤なしの有効成分濃度は 0）
       - 前回の洗浄からの日数・運転日数・累積の加重アスファルテン分・薬剤なしの日数と割合、
         洗浄直後の圧力損失（洗浄後 7 運転日の平均）と、それからの上昇率
       - 機器番号・月
    5. 分割と保存
       - 学習用: 学習期間のうち、目的変数（汚れ係数）が欠測でない行（停止日は含まれない）
       - テスト用: テスト期間の全行（停止日も含む。目的変数は null）

前提と注意:
    - 外れ値の判定（前後の日の中央値）・補間（翌日以降の値）・洗浄の判定（停止後の圧力損失）は、
      その日より後の特徴量（目的変数ではない）を使う。テスト期間の特徴量は 1 年分がまとめて
      与えられる前提のため問題にならないが、日々の逐次予測に使う場合は見直しが必要。
    - データの始まり（2019-01-01）から最初の洗浄までは、前回の洗浄の時期が分からないため、
      洗浄からの日数などの特徴量は null にする。

Usage:
    uv run python scripts/preprocess_heat_exchanger_dummy.py
    uv run python scripts/preprocess_heat_exchanger_dummy.py --output-dir data/processed/dummy_v2

出力:
    - `data/processed/dummy/{train,test}_dataset.parquet`（`--output-dir` で変更可）
    - `outputs/tables/dummy_hx_prep__*.csv`: クリーニングの件数・推定した洗浄・答え合わせ・
      データセットの列の一覧・特徴量と目的変数の相関
    - `outputs/figures/dummy_hx_prep__*.png`: 洗浄の推定とクリーニングの確認図
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# scripts/ から直接実行しても src/ 配下のパッケージをimportできるようにする
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import polars as pl
from eda_heat_exchanger_dummy import (
    OUTLIER_THRESHOLD,
    OUTLIER_WINDOW,
    RATIO_COLUMNS,
    SENSOR_COLUMNS,
    STOPPED,
    TARGET,
    TIME_COL,
    UNIT_COL,
    load_dummy,
    save_figure,
    setup_style,
    to_long,
    weighted_crude_properties,
)

from feature_engineering.datetime_features import DatetimeFeaturesExtractor
from feature_engineering.time_series import MovingAverageTransformer
from util.paths import data_dir, ensure_parent_dir, get_repo_root, outputs_dir
from util.plotting import add_caption

# --- 定数 --------------------------------------------------------------------------

# 入力ディレクトリ（data/raw は読み取り専用として扱い、書き込まない）
RAW_DIR = data_dir() / "raw" / "dummy"
# データセットの出力先
PROCESSED_DIR = data_dir() / "processed" / "dummy"
DATASET_NAME = "dummy_hx_prep"  # 表・図のファイル名の先頭に付ける識別名

# 補間する連続欠測の最大日数（これより長い欠測は null のまま残す）
MAX_INTERPOLATION_GAP = 3
# 洗浄の推定: 停止日数がこれ以上、または停止前後の圧力損失の変化率がこれ以下なら洗浄とみなす
CLEANING_MIN_STOP_DAYS = 4
CLEANING_PRESSURE_CHANGE = -0.2
# 停止前後の圧力損失を測る期間（運転日）。日々の測定誤差を中央値でならす
PRESSURE_WINDOW = 7
# 洗浄直後の圧力損失として平均する、洗浄後の運転日数
BASELINE_DAYS = 7
# 圧力損失の移動平均の窓幅（日）
ROLLING_WINDOW = 7

# 薬剤の性状の列名（`chemical.csv` の列 → データセットの列。原油の性状と紛れないようにする）
CHEMICAL_COLUMNS = {
    "有効成分濃度": "薬剤有効成分濃度",
    "ph": "薬剤ph",
    "比重": "薬剤比重",
    "溶解性": "薬剤溶解性",
}
CRUDE_FEATURES = ("加重アスファルテン分", "加重硫黄分", "加重API比重")
CYCLE_FEATURES = (
    "洗浄後日数",
    "洗浄後運転日数",
    "洗浄後累積加重アスファルテン分",
    "洗浄後薬剤なし日数",
    "洗浄後薬剤なし割合",
    "洗浄直後圧力損失",
)
PRESSURE_DROP_MA = f"圧力損失_ma_{ROLLING_WINDOW}"
# 補間・外れ値の除去の印の列
OUTLIER_FLAGS = tuple(f"{c}_外れ値" for c in SENSOR_COLUMNS)
INTERPOLATION_FLAGS = (*[f"{c}_補間" for c in SENSOR_COLUMNS], "原油比率_補間")


# --- 読み込みと結合 --------------------------------------------------------------------


def combine_periods(train_long: pl.DataFrame, test_long: pl.DataFrame) -> pl.DataFrame:
    """学習期間とテスト期間の縦持ちデータをつなげ、`期間` と `停止` の列を加える。

    Args:
        train_long: 学習期間の縦持ちデータ（`to_long` の出力）。
        test_long: テスト期間の縦持ちデータ。

    Returns:
        機器・日付の順に並べた DataFrame。`期間` は `train` / `test`、`停止` は低温側流量が 0 の日。

    Raises:
        ValueError: 2つの期間の日付が重なっている場合。
    """
    if train_long[TIME_COL].max() >= test_long[TIME_COL].min():
        raise ValueError("学習期間とテスト期間の日付が重なっています")
    return (
        pl.concat(
            [
                train_long.with_columns(pl.lit("train").alias("期間")),
                test_long.with_columns(pl.lit("test").alias("期間")),
            ]
        )
        .with_columns(STOPPED.alias("停止"))
        .sort(UNIT_COL, TIME_COL)
    )


# --- クリーニング --------------------------------------------------------------------


def remove_outliers(
    long: pl.DataFrame,
    columns: tuple[str, ...] = SENSOR_COLUMNS,
    window: int = OUTLIER_WINDOW,
    threshold: float = OUTLIER_THRESHOLD,
) -> pl.DataFrame:
    """運転中の日の外れ値を欠測にし、`<列>_外れ値` の印を付ける。

    EDA の `outlier_candidates` と同じ基準（機器ごとに、運転中の日だけで計算した前後 `window` 日の
    移動中央値との相対差 |値 / 中央値 − 1| が `threshold` 以上）。停止日の流量 0 が中央値を
    引き下げないよう、中央値は運転中の行だけで計算する。

    Args:
        long: `combine_periods` の出力（`停止` 列が必要）。
        columns: 対象の列。
        window: 移動中央値の窓幅（行数）。
        threshold: 外れ値とみなす相対差。

    Returns:
        外れ値を null にし、印の列を加えた DataFrame（行数・行の順は変わらない）。
    """
    medians = long.filter(~pl.col("停止")).select(
        TIME_COL,
        UNIT_COL,
        *[
            pl.col(c)
            .rolling_median(window, center=True, min_samples=window // 2)
            .over(UNIT_COL, order_by=TIME_COL)
            .alias(f"_{c}_median")
            for c in columns
        ],
    )
    out = long.join(medians, on=[TIME_COL, UNIT_COL], how="left", maintain_order="left")
    flags = [
        ((pl.col(c) / pl.col(f"_{c}_median") - 1).abs() >= threshold)
        .fill_null(False)
        .alias(f"{c}_外れ値")
        for c in columns
    ]
    out = out.with_columns(flags)
    return out.with_columns(
        [pl.when(pl.col(f"{c}_外れ値")).then(None).otherwise(pl.col(c)).alias(c) for c in columns]
    ).drop([f"_{c}_median" for c in columns])


def interpolate_short_gaps(
    long: pl.DataFrame, columns: tuple[str, ...], max_gap: int = MAX_INTERPOLATION_GAP
) -> pl.DataFrame:
    """運転中の短い欠測（連続 `max_gap` 日以内）を、機器ごとに線形補間する。

    停止期間をまたいだ補間はしない（運転が連続する区間の中だけで補間する）。区間の端の欠測
    （前後どちらかに値がない）は補間できないため null のまま残す。補間した値には
    `<列>_補間` の印を付ける。

    Args:
        long: `停止` 列を持つ縦持ちデータ（機器・日付の順に並んでいること）。
        columns: 補間する列。
        max_gap: 補間する連続欠測の最大日数。

    Returns:
        補間後の DataFrame（行数・行の順は変わらない）。
    """
    stopped = pl.col("停止")
    # 運転・停止が切り替わるごとに区間番号を振る（機器ごと）
    segment = (stopped != stopped.shift(1)).fill_null(True).cum_sum().over(UNIT_COL)
    out = long.with_columns(segment.alias("_segment"))
    exprs = []
    flags = []
    for c in columns:
        missing = pl.col(c).is_null()
        # 連続する欠測のかたまりの長さ（区間の中で、欠測・非欠測が切り替わるごとに番号を振る）
        run_id = (missing != missing.shift(1)).fill_null(True).cum_sum().over(UNIT_COL, "_segment")
        gap = pl.len().over(UNIT_COL, "_segment", run_id)
        interpolated = pl.col(c).interpolate().over(UNIT_COL, "_segment")
        fill = missing & ~stopped & (gap <= max_gap) & interpolated.is_not_null()
        exprs.append(pl.when(fill).then(interpolated).otherwise(pl.col(c)).alias(c))
        flags.append(fill.alias(f"{c}_補間"))
    return out.with_columns(*exprs, *flags).drop("_segment")


def clean(long: pl.DataFrame) -> pl.DataFrame:
    """外れ値の除去 → 短い欠測の補間 → 圧力損失の再計算をまとめて行う。

    原油比率は3列そろって欠けるため、補間の印は `原油比率_補間` の1列にまとめる
    （両端の値が A + B + C = 1 を満たせば、線形補間した値も合計 1 になる）。

    Args:
        long: `combine_periods` の出力。

    Returns:
        クリーニング後の DataFrame（印の列 `<列>_外れ値` ・ `<列>_補間` ・ `原油比率_補間` 付き）。
    """
    out = remove_outliers(long)
    out = interpolate_short_gaps(out, (*SENSOR_COLUMNS, *RATIO_COLUMNS))
    first, *rest = RATIO_COLUMNS
    return (
        out.rename({f"{first}_補間": "原油比率_補間"})
        .drop([f"{c}_補間" for c in rest])
        .with_columns((pl.col("入口圧力") - pl.col("出口圧力")).alias("圧力損失"))
    )


def cleaning_summary(before: pl.DataFrame, after: pl.DataFrame) -> pl.DataFrame:
    """列ごとの、外れ値として除いた件数・補間した件数と、運転中の欠測件数（前後）を集計する。

    Args:
        before: クリーニング前（`combine_periods` の出力）。
        after: クリーニング後（`clean` の出力）。

    Returns:
        1行 = 1列。列 `column`, `n_outliers`, `n_interpolated`, `missing_before`, `missing_after`
        （欠測件数は運転中の日のみ。学習・テスト期間の合計）。
    """
    rows = []
    for c in (*SENSOR_COLUMNS, *RATIO_COLUMNS, "圧力損失"):
        outlier_flag = f"{c}_外れ値"
        interp_flag = f"{c}_補間" if c in SENSOR_COLUMNS else "原油比率_補間"
        rows.append(
            {
                "column": c,
                "n_outliers": int(after[outlier_flag].sum())
                if outlier_flag in after.columns
                else 0,
                "n_interpolated": int(after[interp_flag].sum()) if c != "圧力損失" else 0,
                "missing_before": before.filter(~pl.col("停止"))[c].null_count(),
                "missing_after": after.filter(~pl.col("停止"))[c].null_count(),
            }
        )
    return pl.DataFrame(rows)


# --- 洗浄の推定 ----------------------------------------------------------------------


def stop_periods(long: pl.DataFrame) -> pl.DataFrame:
    """機器ごとの停止期間（連続する停止日）を一覧にする。

    Args:
        long: `停止` 列を持つ縦持ちデータ（機器・日付の順に並んでいること）。

    Returns:
        1行 = 1停止期間。列 `機器`, `開始日`, `終了日`, `停止日数`。
    """
    stopped = pl.col("停止")
    return (
        long.with_columns(
            (stopped != stopped.shift(1)).fill_null(True).cum_sum().over(UNIT_COL).alias("_run")
        )
        .filter(stopped)
        .group_by(UNIT_COL, "_run")
        .agg(
            pl.col(TIME_COL).min().alias("開始日"),
            pl.col(TIME_COL).max().alias("終了日"),
            pl.len().alias("停止日数"),
        )
        .drop("_run")
        .sort(UNIT_COL, "開始日")
    )


def detect_cleanings(
    long: pl.DataFrame,
    min_stop_days: int = CLEANING_MIN_STOP_DAYS,
    pressure_change: float = CLEANING_PRESSURE_CHANGE,
    window: int = PRESSURE_WINDOW,
) -> pl.DataFrame:
    """停止期間ごとに、洗浄をしたかどうかを運転データから推定する。

    洗浄すると汚れが落ちて圧力損失が下がる。停止前の `window` 運転日と停止後の `window` 運転日の
    圧力損失の中央値を比べ、変化率が `pressure_change` 以下、または停止日数が `min_stop_days`
    以上なら洗浄とみなす（突発停止は短く、圧力損失はほとんど変わらない）。データの端で前後の
    圧力損失が分からない停止期間は、停止日数だけで判定する。

    Args:
        long: クリーニング後の縦持ちデータ（`停止` ・ `圧力損失` 列が必要）。
        min_stop_days: 洗浄とみなす停止日数の下限。
        pressure_change: 洗浄とみなす圧力損失の変化率（停止後 / 停止前 − 1）の上限。
        window: 停止前後の圧力損失を測る運転日数。

    Returns:
        `stop_periods` の出力に、列 `停止前圧力損失`, `停止後圧力損失`, `圧力損失変化率`,
        `洗浄推定`（bool）を加えた表。
    """
    dp = pl.col("圧力損失")
    running = (
        long.filter(~pl.col("停止"))
        .select(
            TIME_COL,
            UNIT_COL,
            # 当日までの window 運転日の中央値（停止前の値として使う）
            dp.rolling_median(window, min_samples=window // 2)
            .over(UNIT_COL, order_by=TIME_COL)
            .alias("停止前圧力損失"),
            # 当日からの window 運転日の中央値（停止後の値として使う）
            dp.rolling_median(window, min_samples=window // 2)
            .shift(-(window - 1))
            .over(UNIT_COL, order_by=TIME_COL)
            .alias("停止後圧力損失"),
        )
        .sort(TIME_COL)
    )
    periods = stop_periods(long)
    before = periods.sort("開始日").join_asof(
        running.select(TIME_COL, UNIT_COL, "停止前圧力損失"),
        left_on="開始日",
        right_on=TIME_COL,
        by=UNIT_COL,
        strategy="backward",
        check_sortedness=False,  # 日付順に並べ済み（by 指定時は確認できないため省く）
    )
    after = before.sort("終了日").join_asof(
        running.select(TIME_COL, UNIT_COL, "停止後圧力損失"),
        left_on="終了日",
        right_on=TIME_COL,
        by=UNIT_COL,
        strategy="forward",
        check_sortedness=False,
    )
    change = pl.col("停止後圧力損失") / pl.col("停止前圧力損失") - 1
    return (
        after.select(UNIT_COL, "開始日", "終了日", "停止日数", "停止前圧力損失", "停止後圧力損失")
        .with_columns(change.alias("圧力損失変化率"))
        .with_columns(
            (
                (pl.col("停止日数") >= min_stop_days)
                | (pl.col("圧力損失変化率") <= pressure_change).fill_null(False)
            ).alias("洗浄推定")
        )
        .sort(UNIT_COL, "開始日")
    )


def cleaning_detection_check(cleanings: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """推定した洗浄を、学習期間の停止・洗浄の記録（`events.csv`）と突き合わせる。

    停止期間の開始日と機器が一致する記録の種類ごとに、洗浄と推定した件数を数える。
    `洗浄停止` ・ `定期修理（洗浄）` はすべて洗浄、`突発停止` はすべて洗浄なしと推定されれば正しい。

    Args:
        cleanings: `detect_cleanings` の出力。
        events: `events.csv`（列 `機器`, `開始日`, `種類`）。

    Returns:
        1行 = 記録の種類（記録のない停止期間は `記録なし`）。
        列 `n_periods`, `n_detected_as_cleaning`。
    """
    period_end = events["終了日"].max()
    return (
        cleanings.filter(pl.col("開始日") <= period_end)
        .join(
            events.select(pl.col("機器").alias(UNIT_COL), "開始日", "種類"),
            on=[UNIT_COL, "開始日"],
            how="left",
        )
        .with_columns(pl.col("種類").fill_null("記録なし"))
        .group_by("種類")
        .agg(
            pl.len().alias("n_periods"),
            pl.col("洗浄推定").sum().alias("n_detected_as_cleaning"),
        )
        .sort("種類")
    )


# --- 特徴量 --------------------------------------------------------------------------


def add_cycle_features(
    long: pl.DataFrame, cleanings: pl.DataFrame, baseline_days: int = BASELINE_DAYS
) -> pl.DataFrame:
    """前回の洗浄からの経過（洗浄サイクル）に関する特徴量を加える。

    推定した洗浄の停止期間の翌日を、新しいサイクルの初日とする。サイクルの中で:

    - `洗浄回数`: データの始まりからの洗浄の回数（サイクルの番号）
    - `洗浄後日数`: サイクルの初日を 1 とする経過日数（突発停止の日も数える）
    - `洗浄後運転日数`: サイクルの中の運転日数（当日を含む）
    - `洗浄後累積加重アスファルテン分`: 運転日の加重アスファルテン分の累積（汚れのもとの量の目安）
    - `洗浄後薬剤なし日数` ・ `洗浄後薬剤なし割合`: 薬剤なしで運転した日数と、運転日数に対する割合
    - `洗浄直後圧力損失`: サイクルの最初の `baseline_days` 運転日の圧力損失の平均
      （当日までの値だけで計算する）

    最初の洗浄より前（`洗浄回数` 0）は前回の洗浄の時期が分からないため、これらは null にする
    （`洗浄回数` は 0 のまま残す）。

    Args:
        long: クリーニング後の縦持ちデータ（`加重アスファルテン分` が必要）。
            機器・日付の順に並んでいること。
        cleanings: `detect_cleanings` の出力。
        baseline_days: 洗浄直後の圧力損失として平均する運転日数。

    Returns:
        特徴量を加えた DataFrame（行数・行の順は変わらない）。
    """
    starts = cleanings.filter("洗浄推定").select(
        UNIT_COL, pl.col("終了日").dt.offset_by("1d").alias(TIME_COL), pl.lit(True).alias("_start")
    )
    out = long.join(starts, on=[UNIT_COL, TIME_COL], how="left", maintain_order="left")
    out = out.with_columns(
        pl.col("_start").fill_null(False).cum_sum().over(UNIT_COL).cast(pl.Int32).alias("洗浄回数")
    )
    cycle = (UNIT_COL, "洗浄回数")
    running = ~pl.col("停止")
    running_days = running.cast(pl.Int32).cum_sum().over(cycle)
    # 洗浄直後の運転日の圧力損失だけを残し、その累積平均を取る（当日より後の値は使わない）
    early = pl.when(running & (running_days <= baseline_days)).then(pl.col("圧力損失"))
    baseline = early.fill_null(0).cum_sum().over(cycle) / early.is_not_null().cum_sum().over(cycle)
    features = [
        ((pl.col(TIME_COL) - pl.col(TIME_COL).min()).dt.total_days() + 1)
        .over(cycle)
        .alias("洗浄後日数"),
        running_days.alias("洗浄後運転日数"),
        # 原油比率が欠測の運転日は、直前の運転日の値で埋めて数える
        pl.when(running)
        .then(pl.col("加重アスファルテン分").fill_null(strategy="forward"))
        .otherwise(0.0)
        .cum_sum()
        .over(cycle)
        .alias("洗浄後累積加重アスファルテン分"),
        (running & (pl.col("薬剤種類") == "なし"))
        .cast(pl.Int32)
        .cum_sum()
        .over(cycle)
        .alias("洗浄後薬剤なし日数"),
        baseline.alias("洗浄直後圧力損失"),
    ]
    out = out.with_columns(features).with_columns(
        (pl.col("洗浄後薬剤なし日数") / pl.col("洗浄後運転日数")).alias("洗浄後薬剤なし割合")
    )
    # 最初の洗浄より前は、前回の洗浄からの経過が分からない
    first_cycle = pl.col("洗浄回数") == 0
    return out.with_columns(
        [pl.when(first_cycle).then(None).otherwise(pl.col(c)).alias(c) for c in CYCLE_FEATURES]
    ).drop("_start")


def build_features(
    cleaned: pl.DataFrame,
    cleanings: pl.DataFrame,
    crude: pl.DataFrame,
    chemical: pl.DataFrame,
) -> pl.DataFrame:
    """クリーニング後のデータに特徴量を加え、列を並べる。

    Args:
        cleaned: `clean` の出力（機器・日付の順）。
        cleanings: `detect_cleanings` の出力。
        crude: `crude.csv`。
        chemical: `chemical.csv`。

    Returns:
        特徴量を加えた DataFrame（日付・機器の順）。
    """
    chem = chemical.select(
        "薬剤種類", *[pl.col(src).alias(dst) for src, dst in CHEMICAL_COLUMNS.items()]
    )
    out = weighted_crude_properties(cleaned, crude)
    out = out.join(chem, on="薬剤種類", how="left", maintain_order="left").with_columns(
        # 薬剤なしは有効成分を入れていないので 0（ほかの性状は該当しないため null のまま）
        pl.col("薬剤有効成分濃度").fill_null(0.0)
    )
    out = add_cycle_features(out, cleanings)
    # 圧力損失の移動平均（機器ごと。日々の測定誤差をならす）。停止日は欠測として扱われる
    out = MovingAverageTransformer(
        "圧力損失", window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 2, group_by=UNIT_COL
    ).fit_transform(out)
    out = DatetimeFeaturesExtractor(TIME_COL, components=("month",)).fit_transform(out)
    out = out.with_columns(
        (pl.col(PRESSURE_DROP_MA) / pl.col("洗浄直後圧力損失") - 1).alias("圧力損失上昇率"),
        pl.col(UNIT_COL).str.slice(2).cast(pl.Int32).alias("機器番号"),
        pl.col("薬剤種類").cast(pl.Categorical),
    ).rename({f"{TIME_COL}_month": "月"})
    columns = [
        TIME_COL,
        UNIT_COL,
        "機器番号",
        "月",
        "期間",
        "停止",
        TARGET,
        *SENSOR_COLUMNS,
        "圧力損失",
        PRESSURE_DROP_MA,
        *RATIO_COLUMNS,
        *CRUDE_FEATURES,
        "薬剤種類",
        *CHEMICAL_COLUMNS.values(),
        "洗浄回数",
        *CYCLE_FEATURES,
        "圧力損失上昇率",
        *OUTLIER_FLAGS,
        *INTERPOLATION_FLAGS,
    ]
    return out.select(columns).sort(TIME_COL, UNIT_COL)


def split_datasets(features: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """学習用・テスト用のデータセットに分ける。

    Args:
        features: `build_features` の出力。

    Returns:
        (学習用: 学習期間で目的変数が欠測でない行, テスト用: テスト期間の全行)。`期間` 列は除く。
    """
    train = features.filter((pl.col("期間") == "train") & pl.col(TARGET).is_not_null())
    test = features.filter(pl.col("期間") == "test")
    return train.drop("期間"), test.drop("期間")


def dataset_columns(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    """データセットの列の一覧（型・欠測率）。

    Returns:
        1行 = 1列。列 `column`, `dtype`, `train_missing_%`, `test_missing_%`。
    """
    return pl.DataFrame(
        {
            "column": train.columns,
            "dtype": [str(t) for t in train.dtypes],
            "train_missing_%": [
                round(train[c].null_count() / train.height * 100, 2) for c in train.columns
            ],
            "test_missing_%": [
                round(test[c].null_count() / test.height * 100, 2) for c in train.columns
            ],
        }
    )


def feature_correlations(train: pl.DataFrame) -> pl.DataFrame:
    """数値の特徴量と目的変数（汚れ係数）の順位相関（学習用データ、全機器・機器ごとの中央値）。

    Returns:
        1行 = 1特徴量。列 `feature`, `spearman_all`, `spearman_median_by_unit`（絶対値の大きい順）。
    """
    numeric = [
        c for c, t in train.schema.items() if t.is_numeric() and c not in (TARGET, "機器番号")
    ]
    overall = train.select([pl.corr(c, TARGET, method="spearman").alias(c) for c in numeric]).row(
        0, named=True
    )
    per_unit = train.group_by(UNIT_COL).agg(
        [pl.corr(c, TARGET, method="spearman").alias(c) for c in numeric]
    )
    return (
        pl.DataFrame(
            {
                "feature": numeric,
                "spearman_all": [overall[c] for c in numeric],
                "spearman_median_by_unit": [per_unit[c].median() for c in numeric],
            }
        )
        .with_columns(pl.col("spearman_all", "spearman_median_by_unit").round(3))
        .sort(pl.col("spearman_all").abs(), descending=True, nulls_last=True)
    )


# --- 図 ------------------------------------------------------------------------------


def plot_cleaning_detection(
    before: pl.DataFrame, features: pl.DataFrame, cleanings: pl.DataFrame, unit: str
) -> plt.Figure:
    """1基の機器について、クリーニング前後の圧力損失・推定した洗浄・汚れ係数を3段で示す。

    Args:
        before: クリーニング前（`combine_periods` の出力）。
        features: `build_features` の出力。
        cleanings: `detect_cleanings` の出力。
        unit: 機器（例: `HX10`）。

    Returns:
        図。
    """
    setup_style()
    raw = before.filter(pl.col(UNIT_COL) == unit)
    feat = features.filter(pl.col(UNIT_COL) == unit)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    axes[0].plot(raw[TIME_COL], raw["圧力損失"], lw=0.6, color="#bbbbbb", label="元の値")
    axes[0].plot(feat[TIME_COL], feat["圧力損失"], lw=0.8, color="#4c72b0", label="クリーニング後")
    # 縦軸はクリーニング後の値の範囲に合わせる（元の値の外れ値ははみ出してよい）
    low, high = feat["圧力損失"].min(), feat["圧力損失"].max()
    if low is not None and high is not None:
        margin = (high - low) * 0.3
        axes[0].set_ylim(low - margin, high + margin)
    axes[0].set_ylabel("圧力損失 (MPa)")
    axes[0].legend(loc="upper left")
    axes[1].plot(feat[TIME_COL], feat["圧力損失上昇率"], lw=0.8, color="#dd8452")
    axes[1].axhline(0, color="gray", lw=0.8)
    axes[1].set_ylabel("圧力損失上昇率")
    axes[2].plot(feat[TIME_COL], feat[TARGET], lw=0.8, color="#55a868")
    axes[2].set_ylabel("汚れ係数 (m²·K/kW)")
    axes[2].set_ylim(bottom=0)
    test_start = features.filter(pl.col("期間") == "test")[TIME_COL].min()
    for row in cleanings.filter(pl.col(UNIT_COL) == unit).iter_rows(named=True):
        color = "#c44e52" if row["洗浄推定"] else "#999999"
        for ax in axes:
            ax.axvspan(row["開始日"], row["終了日"], color=color, alpha=0.4, lw=0)
    for ax in axes:
        ax.axvline(test_start, color="black", lw=1.0, ls="--")
    fig.suptitle(f"{unit}: クリーニングと洗浄の推定（赤: 洗浄と推定した停止、灰: 突発停止と推定）")
    add_caption(
        fig,
        "破線より右はテスト期間（汚れ係数は null）。"
        f"圧力損失上昇率 = {PRESSURE_DROP_MA} / 洗浄直後圧力損失 − 1",
    )
    return fig


# --- メイン処理 ----------------------------------------------------------------------


def _check_output_dir(path: Path) -> None:
    """出力先が data/raw ・ data/external の下でないことを確かめる（元データを上書きしない）。"""
    resolved = path.resolve()
    for protected in (data_dir() / "raw", data_dir() / "external"):
        if resolved.is_relative_to(protected.resolve()):
            raise ValueError(f"{protected} の下には書き込めません: {path}")


def main(argv: list[str] | None = None) -> None:
    """前処理一式を実行し、データセット・表・図を保存する。

    Args:
        argv: コマンドライン引数（Noneなら `sys.argv` を使う。テストから呼ぶ場合に指定）。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=RAW_DIR, help="入力ディレクトリ（既定: data/raw/dummy）"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="データセットの出力先（既定: data/processed/dummy）",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None, help="表・図の出力のルート（既定: outputs/）"
    )
    args = parser.parse_args(argv)
    _check_output_dir(args.output_dir)
    root = args.output_root or outputs_dir()
    figures_dir, tables_dir = root / "figures", root / "tables"

    def save_table(frame: pl.DataFrame, name: str) -> None:
        frame.write_csv(ensure_parent_dir(tables_dir / f"{DATASET_NAME}__{name}.csv"))

    # 1. 読み込みと縦持ちへの変換、学習・テスト期間の結合
    frames = load_dummy(args.raw_dir)
    combined = combine_periods(to_long(frames["train"]), to_long(frames["test"]))
    print(f"=== 縦持ち（学習＋テスト）: {combined.height:,}行 ===")

    # 2. クリーニング
    cleaned = clean(combined)
    summary = cleaning_summary(combined, cleaned)
    save_table(summary, "cleaning_summary")
    print("=== クリーニング（運転中の日の件数。学習＋テスト） ===")
    print(summary)

    # 3. 洗浄の推定と答え合わせ（学習期間の記録）
    cleanings = detect_cleanings(cleaned)
    check = cleaning_detection_check(cleanings, frames["events"])
    save_table(cleanings, "stop_periods")
    save_table(check, "cleaning_detection_check")
    print(
        f"=== 停止期間 {cleanings.height}件（うち洗浄と推定 {cleanings['洗浄推定'].sum()}件） ==="
    )
    print(check)

    # 4. 特徴量
    features = build_features(cleaned, cleanings, frames["crude"], frames["chemical"])

    # 5. 分割と保存
    train, test = split_datasets(features)
    for name, frame in (("train_dataset", train), ("test_dataset", test)):
        path = ensure_parent_dir(args.output_dir / f"{name}.parquet")
        frame.write_parquet(path)
        print(f"保存: {path}（{frame.height:,}行 × {frame.width}列）")
    columns = dataset_columns(train, test)
    correlations = feature_correlations(train)
    save_table(columns, "dataset_columns")
    save_table(correlations, "feature_correlations")
    print("=== 特徴量と汚れ係数の順位相関（上位） ===")
    print(correlations.head(10))

    unit = cleanings.group_by(UNIT_COL).agg(pl.col("洗浄推定").sum()).sort("洗浄推定")[-1, UNIT_COL]
    path = save_figure(
        plot_cleaning_detection(combined, features, cleanings, unit),
        figures_dir / f"{DATASET_NAME}__cleaning_detection_{unit}.png",
    )
    print(
        f"図: {path.relative_to(get_repo_root()) if path.is_relative_to(get_repo_root()) else path}"
    )


if __name__ == "__main__":
    main()
