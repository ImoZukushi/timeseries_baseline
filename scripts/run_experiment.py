"""YAML設定に従ってモデルのCV学習を実行し、予測・スコアの保存と実験ログの記録を行うスクリプト。

複数の設定ファイル（globパターン可）を指定すると順番に実行する。各実験の出力は
`outputs/experiments/{実験名}/{実行日時}/` に保存され、MLflow（`mlruns/`）にも記録される。

Usage:
    uv run python scripts/run_experiment.py --config configs/experiments/lgbm_baseline.yaml
    uv run python scripts/run_experiment.py --config "configs/experiments/*.yaml"
    uv run python scripts/run_experiment.py --config configs/experiments/lgbm.yaml --no-tracking
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

# SHAP等の図をファイル保存のみ行うため非対話型バックエンドに固定する
matplotlib.use("Agg")

from modeling.config import ExperimentConfig, load_experiment_config
from modeling.experiment import run_experiment
from modeling.tracking import MLflowTracker, NullTracker, Tracker


def expand_config_paths(patterns: list[str]) -> list[Path]:
    """設定ファイルのパス・globパターンを展開する（重複は除き、指定順を保つ）。

    Raises:
        FileNotFoundError: どのファイルにも一致しないパターンがある場合。
    """
    paths: list[Path] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matched or not all(Path(m).is_file() for m in matched):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {pattern}")
        for m in matched:
            path = Path(m)
            if path not in paths:
                paths.append(path)
    return paths


def make_tracker(config: ExperimentConfig, enabled: bool) -> Tracker:
    """設定とCLI引数から実験ログの記録先を作る。"""
    if not (enabled and config.tracking.enabled):
        return NullTracker()
    return MLflowTracker(
        experiment_name=config.tracking.experiment_name or config.name,
        tracking_uri=config.tracking.tracking_uri,
    )


def main(argv: list[str] | None = None) -> None:
    """設定ファイルごとに実験を実行し、スコアを表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", nargs="+", required=True, help="実験設定YAMLのパス（globパターン可）"
    )
    parser.add_argument("--no-tracking", action="store_true", help="MLflowに記録しない")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="出力のルートディレクトリ（既定: outputs/experiments）",
    )
    args = parser.parse_args(argv)

    # 実行前に全設定を検証し、途中の設定ミスで長時間の実行が無駄にならないようにする
    configs = [(p, load_experiment_config(p)) for p in expand_config_paths(args.config)]
    for path, config in configs:
        start = time.perf_counter()
        print(f"=== {config.name} ({path}) ===")
        result = run_experiment(
            config,
            tracker=make_tracker(config, enabled=not args.no_tracking),
            output_root=args.output_root,
        )
        cv = result.cv_result
        for metric in config.metrics:
            print(
                f"  {metric}: cv_mean={cv.mean_scores()[metric]:.6f} "
                f"(std={cv.std_scores()[metric]:.6f}) oof={cv.oof_scores[metric]:.6f}"
            )
        print(f"  出力: {result.output_dir}")
        if result.run_id is not None:
            print(f"  MLflow run_id: {result.run_id}")
        print(f"  経過時間: {time.perf_counter() - start:.1f}秒")


if __name__ == "__main__":
    main()
