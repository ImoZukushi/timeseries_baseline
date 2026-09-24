"""modeling.explain のテスト。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from modeling.config import ExperimentConfig
from modeling.experiment import prepare_dataset, run_experiment
from modeling.explain import compute_oof_shap, sample_oof_rows, save_shap_outputs
from modeling.trainer import run_cv

_TARGETS = {"regression": "y_reg", "binary": "y_bin", "multiclass": "y_multi"}
_METRICS = {"regression": ["rmse"], "binary": ["auc"], "multiclass": ["logloss"]}


def _config(
    task: str = "regression", model: str = "lightgbm", **overrides: Any
) -> ExperimentConfig:
    target = _TARGETS[task]
    raw: dict[str, Any] = {
        "name": f"shap_{model}_{task}",
        "task": task,
        "data": {
            "train_path": "unused.csv",
            "target": target,
            "drop_cols": ["id", "cat", "g", "ts", *[t for t in _TARGETS.values() if t != target]],
        },
        "cv": {"method": "kfold", "n_splits": 3},
        "model": {"name": model, "params": {"n_estimators": 30} if model != "linear" else {}},
        "metrics": _METRICS[task],
    }
    return ExperimentConfig.model_validate(raw | overrides)


def _run(frame: pl.DataFrame, cfg: ExperimentConfig) -> tuple[Any, Any]:
    ds = prepare_dataset(cfg, frame)
    return ds, run_cv(cfg, ds.X, ds.y, ds.folds, n_classes=ds.n_classes)


def test_sample_oof_rows_limits_and_skips_unpredicted() -> None:
    fold_ids = np.array([-1, -1, 0, 0, 1, 1, 2, 2])
    assert sample_oof_rows(fold_ids, 100, seed=0).tolist() == [2, 3, 4, 5, 6, 7]
    sampled = sample_oof_rows(fold_ids, 3, seed=0)
    assert len(sampled) == 3
    assert (fold_ids[sampled] >= 0).all()
    assert sampled.tolist() == sorted(sampled.tolist())


@pytest.mark.parametrize("model", ["lightgbm", "xgboost"])
def test_tree_shap_is_additive_for_regression(synthetic_frame: pl.DataFrame, model: str) -> None:
    ds, result = _run(synthetic_frame, _config("regression", model))
    shap_result = compute_oof_shap(_config("regression", model), ds.X, ds.folds, result)
    assert shap_result.explainer_kind == "tree"
    assert shap_result.values.shape == (synthetic_frame.height, ds.X.width)
    reconstructed = shap_result.values.sum(axis=1) + shap_result.base_values
    assert reconstructed == pytest.approx(result.oof_pred[shap_result.rows], abs=1e-4)


def test_tree_shap_binary_is_in_log_odds(synthetic_frame: pl.DataFrame) -> None:
    cfg = _config("binary")
    ds, result = _run(synthetic_frame, cfg)
    shap_result = compute_oof_shap(cfg, ds.X, ds.folds, result)
    assert shap_result.values.ndim == 2
    logit = shap_result.values.sum(axis=1) + shap_result.base_values
    assert 1 / (1 + np.exp(-logit)) == pytest.approx(result.oof_pred[shap_result.rows], abs=1e-5)


def test_tree_shap_multiclass_has_class_axis(synthetic_frame: pl.DataFrame) -> None:
    cfg = _config("multiclass")
    ds, result = _run(synthetic_frame, cfg)
    shap_result = compute_oof_shap(cfg, ds.X, ds.folds, result, n_classes=ds.n_classes)
    assert shap_result.values.shape == (synthetic_frame.height, ds.X.width, 3)
    assert shap_result.base_values.shape == (synthetic_frame.height, 3)
    importance = shap_result.importance(["low", "mid", "high"])
    assert {"mean_abs_shap_low", "mean_abs_shap_mid", "mean_abs_shap_high"} <= set(
        importance.columns
    )


def test_permutation_shap_is_additive_for_non_tree_model(synthetic_frame: pl.DataFrame) -> None:
    cfg = _config("regression", "linear", explain={"max_samples": 30})
    ds, result = _run(synthetic_frame, cfg)
    shap_result = compute_oof_shap(cfg, ds.X, ds.folds, result)
    assert shap_result.explainer_kind == "permutation"
    assert len(shap_result.rows) == 30
    reconstructed = shap_result.values.sum(axis=1) + shap_result.base_values
    assert reconstructed == pytest.approx(result.oof_pred[shap_result.rows], abs=1e-6)


def test_shap_importance_ranks_signal_feature_first(synthetic_frame: pl.DataFrame) -> None:
    # y_reg = 2a + b + ノイズ なので a が最も重要
    ds, result = _run(synthetic_frame, _config())
    importance = compute_oof_shap(_config(), ds.X, ds.folds, result).importance()
    assert importance["feature"].to_list()[:2] == ["a", "b"]


def test_shap_skips_rows_not_predicted_by_time_series_cv(synthetic_frame: pl.DataFrame) -> None:
    raw = _config().model_dump(mode="json", by_alias=True)
    raw.update(task="time_series", cv={"method": "time_series", "n_splits": 3})
    raw["data"]["time_col"] = "ts"
    cfg = ExperimentConfig.model_validate(raw)
    ds, result = _run(synthetic_frame, cfg)
    shap_result = compute_oof_shap(cfg, ds.X, ds.folds, result)
    assert (result.fold_ids[shap_result.rows] >= 0).all()
    assert len(shap_result.rows) == int((result.fold_ids >= 0).sum())


def test_save_shap_outputs_multiclass(synthetic_frame: pl.DataFrame, tmp_path: Path) -> None:
    cfg = _config("multiclass")
    ds, result = _run(synthetic_frame, cfg)
    shap_result = compute_oof_shap(cfg, ds.X, ds.folds, result, n_classes=3)
    paths = save_shap_outputs(shap_result, tmp_path, "多クラス実験", ["0", "1", "2"])
    names = {p.name for p in paths}
    assert names == {
        "shap_importance.csv",
        "shap_importance_bar.png",
        "shap_beeswarm_class_0.png",
        "shap_beeswarm_class_1.png",
        "shap_beeswarm_class_2.png",
        "shap_values.parquet",
    }
    assert all(p.is_file() for p in paths)
    values = pl.read_parquet(tmp_path / "shap_values.parquet")
    assert values.columns[0] == "row"
    assert values.width == 1 + ds.X.width * 3


def test_run_experiment_writes_shap_outputs(synthetic_frame: pl.DataFrame, tmp_path: Path) -> None:
    cfg = _config()
    ds = prepare_dataset(cfg, synthetic_frame)
    result = run_experiment(cfg, ds, output_root=tmp_path)
    shap_dir = result.output_dir / "shap"
    assert (shap_dir / "shap_beeswarm.png").is_file()
    assert (shap_dir / "shap_importance.csv").is_file()
    no_shap = run_experiment(cfg, ds, output_root=tmp_path, explain=False)
    assert not (no_shap.output_dir / "shap").exists()
