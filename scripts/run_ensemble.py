"""YAML設定に従って複数実験の予測をアンサンブルするスクリプト。

各実験（`scripts/run_experiment.py` の出力）のOOF予測・テスト予測を読み込み、
設定した方法（mean / rank_mean / weighted / stacking）で統合する。結果は
`outputs/ensembles/{アンサンブル名}/{実行日時}/` に保存され、MLflowにも記録される。

Usage:
    uv run python scripts/run_ensemble.py --config configs/ensembles/example_blend.yaml
    uv run python scripts/run_ensemble.py --config configs/ensembles/blend.yaml --no-tracking
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modeling.config import load_ensemble_config
from modeling.ensemble import run_ensemble
from modeling.tracking import MLflowTracker, NullTracker, Tracker


def main(argv: list[str] | None = None) -> None:
    """アンサンブルを実行し、構成要素とアンサンブルのスコアを表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="アンサンブル設定YAMLのパス")
    parser.add_argument("--no-tracking", action="store_true", help="MLflowに記録しない")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="出力のルートディレクトリ（既定: outputs/ensembles）",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=None,
        help="構成要素の実験結果を探すルート（既定: outputs/experiments）",
    )
    args = parser.parse_args(argv)

    config = load_ensemble_config(args.config)
    tracker: Tracker = NullTracker()
    if not args.no_tracking and config.tracking.enabled:
        tracker = MLflowTracker(
            experiment_name=config.tracking.experiment_name or config.name,
            tracking_uri=config.tracking.tracking_uri,
        )
    result = run_ensemble(
        config,
        tracker=tracker,
        output_root=args.output_root,
        experiments_root=args.experiments_root,
    )
    print(f"=== {config.name} (method={config.method}) ===")
    print(result.scores)
    if result.weights is not None:
        print("  重み: " + ", ".join(f"{k}={v:.4f}" for k, v in result.weights.items()))
    print(f"  出力: {result.output_dir}")
    if result.run_id is not None:
        print(f"  MLflow run_id: {result.run_id}")


if __name__ == "__main__":
    main()
