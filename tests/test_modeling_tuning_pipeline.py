"""modeling.tuning_pipeline（探索 → 学習 → アンサンブル）と scripts/run_pipeline.py のテスト。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from pandas_helpers import as_list, row_at
from pydantic import ValidationError

from modeling.config import load_pipeline_config
from modeling.tuning_pipeline import load_members, run_tuning_pipeline, validate_members

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def pipeline_dir(synthetic_frame: pd.DataFrame, tmp_path: Path) -> Path:
    """合成データで小さく回せるパイプライン（LightGBM を探索、linear は探索なし）。"""
    train = tmp_path / "train.parquet"
    synthetic_frame.to_parquet(train, index=False)
    root = tmp_path / "configs"
    _write_yaml(
        root / "members" / "common.yaml",
        {
            "task": "regression",
            "data": {
                "train_path": str(train),
                "target": "y_reg",
                "drop_cols": ["id", "cat", "g", "ts", "y_bin", "y_multi"],
            },
            "cv": {"method": "kfold", "n_splits": 3, "seed": 0},
            "metrics": ["rmse", "mae"],
            "explain": {"enabled": False},
            "evaluation": {"enabled": False},
        },
    )
    _write_yaml(
        root / "members" / "lgbm.yaml",
        {
            "base": "common.yaml",
            "name": "pipe_lgbm",
            "model": {"name": "lightgbm", "params": {"n_estimators": 20}},
            "tuning": {
                "search_space": {
                    "learning_rate": {"type": "float", "low": 0.05, "high": 0.3, "log": True}
                }
            },
        },
    )
    _write_yaml(
        root / "members" / "linear.yaml",
        {"base": "common.yaml", "name": "pipe_linear", "model": {"name": "linear"}},
    )
    _write_yaml(
        root / "pipe.yaml",
        {
            "name": "pipe",
            "members": [
                {"config": "members/lgbm.yaml", "n_trials": 2},
                {"config": "members/linear.yaml", "tune": False},
            ],
            "ensemble": {"methods": ["mean", "weighted"]},
            "tracking": {"enabled": False},
        },
    )
    return root


# --- 設定 ------------------------------------------------------------------------------------


def test_load_pipeline_config_resolves_member_paths(pipeline_dir: Path) -> None:
    config = load_pipeline_config(pipeline_dir / "pipe.yaml")
    assert config.members[0].config == (pipeline_dir / "members" / "lgbm.yaml").resolve()
    assert config.members[1].tune is False
    members = load_members(config)
    assert [m.config.name for m in members] == ["pipe_lgbm", "pipe_linear"]
    validate_members(config, members)


def test_inherited_members_are_relative_to_base_file(pipeline_dir: Path, tmp_path: Path) -> None:
    # 別のディレクトリから base で継承しても、members のパスは土台のファイルが基準
    child = _write_yaml(
        tmp_path / "elsewhere" / "child.yaml",
        {"base": str(pipeline_dir / "pipe.yaml"), "name": "child"},
    )
    config = load_pipeline_config(child)
    assert config.name == "child"
    assert config.members[0].config == (pipeline_dir / "members" / "lgbm.yaml").resolve()


def test_validate_members_rejects_inconsistent_members(pipeline_dir: Path) -> None:
    # CV分割が違う構成モデルはアンサンブルできない
    _write_yaml(
        pipeline_dir / "members" / "other_cv.yaml",
        {"base": "lgbm.yaml", "name": "pipe_other", "cv": {"n_splits": 4}},
    )
    _write_yaml(
        pipeline_dir / "bad.yaml",
        {
            "name": "bad",
            "members": [{"config": "members/lgbm.yaml"}, {"config": "members/other_cv.yaml"}],
        },
    )
    config = load_pipeline_config(pipeline_dir / "bad.yaml")
    with pytest.raises(ValueError, match="cv"):
        validate_members(config, load_members(config))


def test_validate_members_rejects_duplicate_names_and_rank_mean(pipeline_dir: Path) -> None:
    _write_yaml(
        pipeline_dir / "dup.yaml",
        {"name": "dup", "members": [{"config": "members/lgbm.yaml"}] * 2},
    )
    config = load_pipeline_config(pipeline_dir / "dup.yaml")
    with pytest.raises(ValueError, match="重複"):
        validate_members(config, load_members(config))
    _write_yaml(
        pipeline_dir / "rank.yaml",
        {
            "name": "rank",
            "members": [{"config": "members/lgbm.yaml"}, {"config": "members/linear.yaml"}],
            "ensemble": {"methods": ["rank_mean"]},
        },
    )
    config = load_pipeline_config(pipeline_dir / "rank.yaml")
    with pytest.raises(ValidationError, match="rank_mean"):
        validate_members(config, load_members(config))


def test_pipeline_config_rejects_duplicate_methods(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "p.yaml",
        {"name": "p", "members": [{"config": "x.yaml"}], "ensemble": {"methods": ["mean", "mean"]}},
    )
    with pytest.raises(ValidationError, match="重複"):
        load_pipeline_config(path)


def test_bundled_example_pipeline_is_valid() -> None:
    config = load_pipeline_config(REPO_ROOT / "configs" / "pipelines" / "example_ensemble.yaml")
    members = load_members(config)
    validate_members(config, members)
    assert [m.config.model.name for m in members] == ["lightgbm", "xgboost", "linear"]
    # 探索範囲は configs/search_spaces から継承される
    assert all(m.config.tuning.search_space for m in members)
    assert set(config.ensemble.methods) == {"mean", "weighted", "stacking"}


# --- 実行 ------------------------------------------------------------------------------------


def test_run_tuning_pipeline_end_to_end(pipeline_dir: Path, tmp_path: Path) -> None:
    config = load_pipeline_config(pipeline_dir / "pipe.yaml")
    logs: list[str] = []
    result = run_tuning_pipeline(
        config, output_root=tmp_path / "out", optuna_dir=tmp_path / "optuna", log=logs.append
    )
    out = result.output_dir
    for name in ("members.csv", "best_params.yaml", "summary.csv", "final.yaml", "pipeline.yaml"):
        assert (out / name).is_file(), name

    members = pd.read_csv(out / "members.csv")
    assert members["member"].pipe(as_list) == ["pipe_lgbm", "pipe_linear"]
    assert members["tuned"].pipe(as_list) == [True, False]
    assert members["n_new_trials"][0] == 2
    # アンサンブルの構成要素は今回の実行結果（最新の実行を探さない）
    for ensemble in result.ensembles.values():
        sources = yaml.safe_load((ensemble.output_dir / "config.yaml").read_text(encoding="utf-8"))
        assert all(str(out) in s for s in sources["sources"].values())

    summary = pd.read_csv(out / "summary.csv")
    assert list(summary[["kind", "name"]].itertuples(index=False, name=None)) == [
        ("member", "pipe_lgbm"),
        ("member", "pipe_linear"),
        ("ensemble", "mean"),
        ("ensemble", "weighted"),
    ]
    best = row_at(summary.sort_values("rank", kind="stable"), 0)
    final = yaml.safe_load((out / "final.yaml").read_text(encoding="utf-8"))
    assert (final["kind"], final["name"]) == (best["kind"], best["name"])
    assert final["score"] == pytest.approx(summary["rmse"].min())
    assert any("最終モデル" in line for line in logs)

    # 同じ設定で再実行すると、探索は目標に達しているので飛ばされる
    again = run_tuning_pipeline(
        config, output_root=tmp_path / "out", optuna_dir=tmp_path / "optuna"
    )
    tr = again.members["pipe_lgbm"].tuning_result
    assert tr is not None and tr.n_new_trials == 0
    assert tr.best_params == result.members["pipe_lgbm"].tuning_result.best_params  # type: ignore[union-attr]


def test_single_member_pipeline_skips_ensemble(pipeline_dir: Path, tmp_path: Path) -> None:
    path = _write_yaml(
        pipeline_dir / "single.yaml",
        {"name": "single", "members": [{"config": "members/linear.yaml", "tune": False}]},
    )
    result = run_tuning_pipeline(load_pipeline_config(path), output_root=tmp_path / "out")
    assert result.ensembles == {}
    assert result.final["name"] == "pipe_linear"


def test_run_pipeline_script(pipeline_dir: Path, tmp_path: Path) -> None:
    import run_pipeline as script

    args = ["--config", str(pipeline_dir / "pipe.yaml"), "--no-tracking"]
    script.main([*args, "--check"])
    assert not (tmp_path / "out").exists()
    script.main(
        [
            *args,
            "--skip-tuning",
            "--output-root",
            str(tmp_path / "out"),
            "--optuna-dir",
            str(tmp_path / "optuna"),
        ]
    )
    members = pd.read_csv(next((tmp_path / "out").rglob("members.csv")))
    assert members["tuned"].pipe(as_list) == [False, False]
