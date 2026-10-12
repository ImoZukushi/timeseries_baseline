"""探索の設定に従って、単一モデルで特徴量の組み合わせ（パターン）を学習・比較するスクリプト。

特徴量をグループに分け、グループの組み合わせ（パターン）ごとに同じ CV 分割で学習して、
基準のパターンとの差（fold 平均と、基準に勝った fold の数）を比べる。設定すれば、あるパターンから
グループを1つずつ外すアブレーションも自動で行う。結果は `outputs/feature_search/<name>/<日時>/` に
まとめて保存する（`summary.csv` ・ `ablation.csv` ・ `feature_importance.csv` ・ 図）。

Usage:
    uv run python scripts/run_feature_search.py --config configs/feature_search/hx_dummy.yaml
    uv run python scripts/run_feature_search.py --config <探索の設定> --check   # 確認のみ
    uv run python scripts/run_feature_search.py --config <探索の設定> --patterns 05_cleaning_cycle
    uv run python scripts/run_feature_search.py --config <探索の設定> --no-ablation --no-tracking
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

# 図はファイル保存のみ行うため非対話型バックエンドに固定する
matplotlib.use("Agg")

import polars as pl  # noqa: E402

from modeling.config import ExperimentConfig, load_feature_search_config  # noqa: E402
from modeling.experiment import load_table  # noqa: E402
from modeling.feature_search import (  # noqa: E402
    build_patterns,
    run_feature_search,
    validate_patterns,
)
from modeling.tracking import Tracker, make_tracker  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    """特徴量の探索を実行し、パターンの比較を表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="探索の設定YAMLのパス")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=None,
        help="実行するパターン名（既定: 全パターン。基準・アブレーションの土台は自動で含める）",
    )
    parser.add_argument("--no-ablation", action="store_true", help="アブレーションを行わない")
    parser.add_argument("--no-tracking", action="store_true", help="MLflowに記録しない")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="出力のルートディレクトリ（既定: outputs/feature_search）",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="設定を検証し、パターンごとのグループと列数を表示するだけで学習しない",
    )
    args = parser.parse_args(argv)

    # 学習の前に全パターンの実験設定を組み立てて検証する（設定ミスで実行が無駄にならないように）
    search = load_feature_search_config(args.config)
    ablation = not args.no_ablation
    patterns = build_patterns(search, only=args.patterns, ablation=ablation)
    print(f"特徴量の探索 {search.name}: {len(patterns)}パターン（基準: {search.baseline_pattern}）")
    if args.check:
        validate_patterns(patterns, load_table(patterns[0].config.data.train_path).columns)
        for p in patterns:
            n_columns = len(p.config.data.feature_cols or [])
            print(
                f"  {p.name}: 列 {n_columns}、ステップ {len(p.config.features)}、"
                f"グループ {p.groups}"
            )
        print("設定の検証のみ: 問題ありません")
        return

    def tracker_factory(cfg: ExperimentConfig) -> Tracker:
        return make_tracker(cfg.tracking, cfg.name, enabled=not args.no_tracking)

    start = time.perf_counter()
    result = run_feature_search(
        search,
        only=args.patterns,
        ablation=ablation,
        tracker_factory=tracker_factory,
        output_root=args.output_root,
        log=print,
    )
    primary = patterns[0].config.primary_metric
    with pl.Config(tbl_rows=60, tbl_cols=20, tbl_width_chars=200, fmt_str_lengths=40):
        print("=== パターンの比較（主指標の良い順） ===")
        print(
            result.summary.select(
                "rank",
                "pattern",
                "n_columns",
                f"cv_mean_{primary}",
                f"cv_std_{primary}",
                f"improvement_{primary}",
                "fold_wins",
                "n_folds",
            )
        )
        if not result.ablation.is_empty():
            print("=== アブレーション（外すと悪くなる量。正: 効いている） ===")
            print(result.ablation)
    print(f"出力: {result.output_dir}")
    print(f"経過時間: {time.perf_counter() - start:.1f}秒")


if __name__ == "__main__":
    main()
