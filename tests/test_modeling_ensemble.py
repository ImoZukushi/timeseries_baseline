"""modeling.ensemble のテスト。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from modeling.config import EnsembleConfig, ExperimentConfig
from modeling.ensemble import (
    MemberPredictions,
    WeightedBlender,
    check_members_consistent,
    cross_fit_blend,
    latest_run_dir,
    load_member,
    run_ensemble,
)
from modeling.experiment import prepare_dataset, run_experiment
from modeling.io import OOF_FILENAME, TEST_FILENAME
from modeling.metrics import get_metric
from modeling.tracking import MLflowTracker

_TARGETS = {"regression": "y_reg", "binary": "y_bin", "multiclass": "y_multi"}
_METRICS = {"regression": ["rmse", "mae"], "binary": ["auc"], "multiclass": ["logloss"]}


def _exp_config(task: str, model: str) -> ExperimentConfig:
    target = _TARGETS[task]
    return ExperimentConfig.model_validate(
        {
            "name": f"{model}_{task}",
            "task": task,
            "data": {
                "train_path": "unused.csv",
                "target": target,
                "id_col": "id",
                "drop_cols": ["cat", "g", "ts", *[t for t in _TARGETS.values() if t != target]],
            },
            "cv": {"method": "kfold", "n_splits": 3},
            "model": {"name": model, "params": {"n_estimators": 30} if model == "lightgbm" else {}},
            "metrics": _METRICS[task],
            "explain": {"enabled": False},
        }
    )


def _ens_config(task: str, method: str, **overrides: Any) -> EnsembleConfig:
    raw: dict[str, Any] = {
        "name": f"ens_{task}_{method}",
        "task": task,
        "members": [
            {"name": "lgbm", "experiment": f"lightgbm_{task}"},
            {"name": "linear", "experiment": f"linear_{task}"},
        ],
        "method": method,
        "metrics": _METRICS[task],
    }
    return EnsembleConfig.model_validate(raw | overrides)


def _run_members(frame: pl.DataFrame, task: str, root: Path) -> None:
    for model in ("lightgbm", "linear"):
        cfg = _exp_config(task, model)
        run_experiment(cfg, prepare_dataset(cfg, frame, frame.head(6)), output_root=root)


def _member(name: str, oof: np.ndarray, folds: np.ndarray, y: np.ndarray) -> MemberPredictions:
    return MemberPredictions(name, oof, folds, y, None, None, source=name)


# --- config -------------------------------------------------------------------------


def test_ensemble_config_validation() -> None:
    with pytest.raises(ValidationError, match="rank_mean"):
        _ens_config("regression", "rank_mean")
    with pytest.raises(ValidationError, match="いずれか1つ"):
        _ens_config(
            "regression",
            "mean",
            members=[{"name": "a", "experiment": "x", "path": "p"}, {"name": "b", "run_id": "r"}],
        )
    with pytest.raises(ValidationError, match="重複"):
        _ens_config(
            "regression",
            "mean",
            members=[{"name": "a", "experiment": "x"}, {"name": "a", "experiment": "y"}],
        )
    with pytest.raises(ValidationError):
        _ens_config("regression", "mean", members=[{"name": "a", "experiment": "x"}])


# --- 部品 -----------------------------------------------------------------------------


def test_weighted_blender_prefers_better_member() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    preds = np.stack([y + rng.normal(size=300) * 0.05, y + rng.normal(size=300) * 2.0])
    blender = WeightedBlender(get_metric("rmse")).fit(preds, y)
    assert blender.weights_ is not None
    assert blender.weights_.sum() == pytest.approx(1.0)
    assert (blender.weights_ >= 0).all()
    assert blender.weights_[0] > 0.95


def test_cross_fit_blend_leaves_unpredicted_rows_nan() -> None:
    y = np.arange(8, dtype=float)
    folds = np.array([-1, -1, 0, 0, 1, 1, 2, 2])
    preds = np.stack([y, y + 1.0])
    oof = cross_fit_blend(lambda: WeightedBlender(get_metric("rmse")), preds, y, folds)
    assert np.isnan(oof[:2]).all()
    assert oof[2:] == pytest.approx(y[2:], abs=1e-2)


def test_check_members_consistent_detects_mismatch() -> None:
    y = np.array([0.0, 1.0, 2.0])
    folds = np.array([0, 1, 1])
    base = _member("a", y, folds, y)
    check_members_consistent([base, _member("b", y + 1, folds, y)])
    with pytest.raises(ValueError, match="CV分割"):
        check_members_consistent([base, _member("b", y, np.array([1, 0, 0]), y)])
    with pytest.raises(ValueError, match="目的変数"):
        check_members_consistent([base, _member("b", y, folds, y + 1)])
    with pytest.raises(ValueError, match="形"):
        check_members_consistent([base, _member("b", np.zeros((3, 2)), folds, y)])


def test_latest_run_dir_picks_newest(tmp_path: Path) -> None:
    for stamp in ("20240101_000000_000000", "20240201_000000_000000"):
        d = tmp_path / "exp" / stamp
        d.mkdir(parents=True)
        (d / OOF_FILENAME).write_bytes(b"")
    (tmp_path / "exp" / "20250101_incomplete").mkdir()  # 予測ファイルが無い実行は無視
    assert latest_run_dir("exp", tmp_path).name == "20240201_000000_000000"
    with pytest.raises(FileNotFoundError):
        latest_run_dir("missing", tmp_path)


# --- 実行 -------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["mean", "weighted", "stacking"])
def test_run_ensemble_regression(
    synthetic_frame: pl.DataFrame, tmp_path: Path, method: str
) -> None:
    _run_members(synthetic_frame, "regression", tmp_path / "exp")
    result = run_ensemble(
        _ens_config("regression", method),
        output_root=tmp_path / "ens",
        experiments_root=tmp_path / "exp",
    )
    scores = result.scores
    assert scores["model"].to_list() == ["lgbm", "linear", f"ensemble_{method}"]
    member_rmse = scores["rmse"].to_list()[:2]
    # アンサンブルは少なくとも悪い方の構成要素よりは良い
    assert scores["rmse"].to_list()[2] <= max(member_rmse)
    assert result.test_pred is not None and result.test_pred.shape == (6,)
    out = result.output_dir
    assert {p.name for p in out.iterdir()} >= {
        "config.yaml",
        "ensemble_scores.csv",
        OOF_FILENAME,
        TEST_FILENAME,
    }
    assert pl.read_parquet(out / TEST_FILENAME)["id"].to_list() == list(range(6))
    if method in ("mean", "weighted"):
        assert result.weights is not None
        assert sum(result.weights.values()) == pytest.approx(1.0)
    else:
        assert result.weights is None


def test_run_ensemble_binary_rank_mean(synthetic_frame: pl.DataFrame, tmp_path: Path) -> None:
    _run_members(synthetic_frame, "binary", tmp_path / "exp")
    result = run_ensemble(
        _ens_config("binary", "rank_mean"),
        output_root=tmp_path / "ens",
        experiments_root=tmp_path / "exp",
    )
    assert result.test_pred is not None
    assert result.test_pred.min() >= 0.0 and result.test_pred.max() <= 1.0
    assert 0.5 < result.scores["auc"].to_list()[-1] <= 1.0


def test_run_ensemble_multiclass_stacking(synthetic_frame: pl.DataFrame, tmp_path: Path) -> None:
    _run_members(synthetic_frame, "multiclass", tmp_path / "exp")
    result = run_ensemble(
        _ens_config("multiclass", "stacking"),
        output_root=tmp_path / "ens",
        experiments_root=tmp_path / "exp",
    )
    assert result.oof_pred.shape == (synthetic_frame.height, 3)
    assert result.oof_pred.sum(axis=1) == pytest.approx(np.ones(synthetic_frame.height))


def test_load_member_from_mlflow_run(synthetic_frame: pl.DataFrame, tmp_path: Path) -> None:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = MLflowTracker("ens_src", tracking_uri=uri, artifact_root=tmp_path / "artifacts")
    cfg = _exp_config("regression", "linear")
    exp = run_experiment(
        cfg, prepare_dataset(cfg, synthetic_frame), tracker=tracker, output_root=tmp_path / "exp"
    )
    assert exp.run_id is not None
    ens = _ens_config(
        "regression",
        "mean",
        members=[
            {"name": "a", "run_id": exp.run_id},
            {"name": "b", "experiment": "linear_regression"},
        ],
    )
    member = load_member(ens.members[0], tracking_uri=uri)
    assert member.source == f"mlflow:{exp.run_id}"
    assert member.oof == pytest.approx(exp.cv_result.oof_pred)
    assert member.test is None  # テストデータ無しの実験


def test_run_ensemble_script(synthetic_frame: pl.DataFrame, tmp_path: Path) -> None:
    import run_ensemble as script

    _run_members(synthetic_frame, "regression", tmp_path / "exp")
    config_path = tmp_path / "ens.yaml"
    config_path.write_text(
        yaml.safe_dump(_ens_config("regression", "weighted").model_dump(mode="json")),
        encoding="utf-8",
    )
    script.main(
        [
            "--config",
            str(config_path),
            "--no-tracking",
            "--output-root",
            str(tmp_path / "ens"),
            "--experiments-root",
            str(tmp_path / "exp"),
        ]
    )
    assert any((tmp_path / "ens" / "ens_regression_weighted").iterdir())
