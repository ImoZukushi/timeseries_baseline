"""パイプラインの設定に従って、各モデルの探索 → 学習 → アンサンブルを一括で実行するスクリプト。

構成モデルごとに Optuna でハイパーパラメータを探索し（試行回数は study 全体の目標で、足りない
分だけ実行する）、最良パラメータで CV 学習した後、指定した複数の統合方法でアンサンブルして比較する。
結果は `outputs/pipelines/<name>/<日時>/` にまとめて保存する（`summary.csv`・`final.yaml` など）。

Usage:
    uv run python scripts/run_pipeline.py --config configs/pipelines/hx_ensemble.yaml
    uv run python scripts/run_pipeline.py --config <パイプライン設定> --no-tracking
    uv run python scripts/run_pipeline.py --config <パイプライン設定> --skip-tuning
    uv run python scripts/run_pipeline.py --config <パイプライン設定> --check   # 設定の検証のみ

中断した場合は同じコマンドで再実行すると、各 study の続きから探索する（目標に達した構成モデルは
探索を飛ばして学習だけを行う）。探索範囲・特徴量・データなどを変えると、自動で新しい study になる。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

# SHAP等の図をファイル保存のみ行うため非対話型バックエンドに固定する
matplotlib.use("Agg")

import polars as pl  # noqa: E402

from modeling.config import ExperimentConfig, load_pipeline_config  # noqa: E402
from modeling.tracking import Tracker, make_tracker  # noqa: E402
from modeling.tuning_pipeline import (  # noqa: E402
    load_members,
    run_tuning_pipeline,
    validate_members,
)


def main(argv: list[str] | None = None) -> None:
    """パイプラインを実行し、構成モデルとアンサンブルの比較を表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="パイプライン設定YAMLのパス")
    parser.add_argument("--no-tracking", action="store_true", help="MLflowに記録しない")
    parser.add_argument(
        "--skip-tuning",
        action="store_true",
        help="全構成モデルを探索せず、設定の params で学習する",
    )
    parser.add_argument(
        "--optuna-dir",
        type=Path,
        default=None,
        help="Optuna studyの保存先ディレクトリ（既定: outputs/optuna）",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="出力のルートディレクトリ（既定: outputs/pipelines）",
    )
    parser.add_argument("--check", action="store_true", help="設定の検証だけを行い、学習しない")
    args = parser.parse_args(argv)

    # 学習の前に、パイプラインと全構成モデルの設定を検証する（長時間の実行が途中で失敗しないように）
    config = load_pipeline_config(args.config)
    members = load_members(config)
    validate_members(config, members)
    print(f"パイプライン {config.name}: 構成モデル {[m.config.name for m in members]}")
    if args.check:
        print("設定の検証のみ: 問題ありません")
        return

    def member_tracker(cfg: ExperimentConfig) -> Tracker:
        return make_tracker(cfg.tracking, cfg.name, enabled=not args.no_tracking)

    ensemble_tracker = make_tracker(config.tracking, config.name, enabled=not args.no_tracking)

    start = time.perf_counter()
    result = run_tuning_pipeline(
        config,
        member_tracker=member_tracker,
        ensemble_tracker=ensemble_tracker,
        output_root=args.output_root,
        optuna_dir=args.optuna_dir,
        skip_tuning=args.skip_tuning,
        log=print,
    )
    with pl.Config(tbl_rows=50, tbl_cols=20, tbl_width_chars=200):
        print(result.summary.sort("rank"))
    print(f"出力: {result.output_dir}")
    print(f"経過時間: {time.perf_counter() - start:.1f}秒")


if __name__ == "__main__":
    main()
