"""data/raw配下の全CSVファイルに対してデータ品質確認・時系列EDAを実施するスクリプト。

各CSVファイルについて、データフレーム全体の品質確認（行数・列数・重複行・欠損値の
全体像・記録パターンの目視確認）とカラム単位の品質確認（型・欠損率・ユニーク数・
記述統計・歪度・上位出現値・相関）を行い、表を `outputs/tables/`、図を
`outputs/figures/` に保存する。全ファイル分のサマリも合わせて保存する。

さらに、日時型の列を持つ（＝時系列データとみなせる）ファイルについては、数値列ごとに
6種類の変換系列（原系列・対数系列・差分系列・対数差分系列・季節差分系列・
季節対数差分系列）に対する時系列診断図（生値と移動平均、ACF/PACFのコレログラム）も
`outputs/figures/` に保存する。

Usage:
    uv run python scripts/run_csv_quality_checks.py
    uv run python scripts/run_csv_quality_checks.py --pattern "wind_0/*.csv"
    uv run python scripts/run_csv_quality_checks.py --moving-average-window 7 --seasonal-period 12
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

# GUIバックエンド（Tk等）だと図を1枚生成するたびにウィンドウ生成コストがかかり
# 非常に遅くなるため、ファイル保存のみを行う非対話型バックエンドに固定する。
# matplotlib.pyplot を最初にimportする前に設定する必要がある。
matplotlib.use("Agg")

import polars as pl
import seaborn as sns

from analysis_project.csv_quality import make_dataset_name, read_csv_auto, run_quality_checks
from analysis_project.paths import data_dir, outputs_dir
from analysis_project.time_series_eda import run_time_series_checks


def main() -> None:
    """data/raw配下のCSVファイルを走査し、品質確認・時系列EDAの結果を出力する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern", default="**/*.csv", help="対象CSVのglobパターン（data/raw基準、既定: 全件）"
    )
    parser.add_argument(
        "--moving-average-window", type=int, default=5, help="移動平均の窓幅（点数、既定: 5）"
    )
    parser.add_argument(
        "--seasonal-period", type=int, default=7, help="季節差分・季節対数差分の周期（既定: 7）"
    )
    parser.add_argument(
        "--nlags", type=int, default=40, help="ACF/PACFで計算・表示するラグ数の上限（既定: 40）"
    )
    args = parser.parse_args()

    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

    raw_dir = data_dir() / "raw"
    figures_dir = outputs_dir() / "figures"
    tables_dir = outputs_dir() / "tables"

    csv_paths = sorted(raw_dir.glob(args.pattern))
    if not csv_paths:
        print(f"対象ファイルが見つかりません: {raw_dir}/{args.pattern}")
        return

    df_overviews: list[pl.DataFrame] = []
    col_overviews: list[pl.DataFrame] = []
    failures: list[tuple[Path, str]] = []
    n_time_series_plots = 0

    for i, path in enumerate(csv_paths, 1):
        start = time.time()
        print(f"[{i}/{len(csv_paths)}] {path.relative_to(raw_dir)} を処理中...", flush=True)
        try:
            dataset_name = make_dataset_name(path, raw_dir)
            df = read_csv_auto(path)

            df_ov, col_ov = run_quality_checks(
                df, dataset_name, str(path.relative_to(raw_dir)), figures_dir, tables_dir
            )
            df_overviews.append(df_ov)
            col_overviews.append(col_ov)

            processed = run_time_series_checks(
                df,
                dataset_name,
                figures_dir,
                moving_average_window=args.moving_average_window,
                seasonal_period=args.seasonal_period,
                nlags=args.nlags,
            )
            n_time_series_plots += len(processed)

            elapsed = time.time() - start
            ts_note = f", 時系列図 {len(processed)}系列" if processed else ""
            print(f"  -> 完了 ({elapsed:.1f}秒, {df_ov['n_rows'][0]:,}行{ts_note})")
        except Exception as e:
            print(f"  -> 失敗: {e}")
            failures.append((path, str(e)))

    if df_overviews:
        summary_df = pl.concat(df_overviews, how="vertical_relaxed")
        summary_df.write_csv(tables_dir / "summary_all_files__dataframe_overview.csv")
    if col_overviews:
        summary_col = pl.concat(col_overviews, how="vertical_relaxed")
        summary_col.write_csv(tables_dir / "summary_all_files__column_overview.csv")

    print(
        f"\n完了: {len(df_overviews)}件成功, {len(failures)}件失敗, "
        f"時系列図 延べ{n_time_series_plots}系列"
    )
    for path, err in failures:
        print(f"  失敗: {path}: {err}")


if __name__ == "__main__":
    main()
