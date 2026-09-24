"""modeling.config / modeling.tasks のテスト（指標は test_modeling_metrics.py）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from modeling.config import ExperimentConfig, load_experiment_config
from modeling.tasks import Task, encode_target, predict


def _config(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "exp",
        "task": "regression",
        "data": {"train_path": "train.csv", "target": "y"},
        "model": {"name": "lightgbm"},
        "metrics": ["rmse"],
    }
    return base | overrides


# --- config -------------------------------------------------------------------


def test_config_defaults() -> None:
    cfg = ExperimentConfig.model_validate(_config())
    assert cfg.cv.method == "kfold"
    assert cfg.primary_metric == "rmse"
    assert cfg.test_prediction == "fold_mean"


def test_config_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_config(unknown_key=1))


def test_config_rejects_unknown_metric() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_config(metrics=["not_a_metric"]))


def test_config_rejects_metric_task_mismatch() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_config(metrics=["auc"]))
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_config(task="binary", metrics=["rmse"]))


@pytest.mark.parametrize("method", ["kfold", "group"])
def test_time_series_task_rejects_non_temporal_cv(method: str) -> None:
    cfg = _config(task="time_series", cv={"method": method})
    cfg["data"]["group_col"] = "g"
    with pytest.raises(ValidationError, match="time_series"):
        ExperimentConfig.model_validate(cfg)


def test_time_series_task_accepts_temporal_cv() -> None:
    cfg = ExperimentConfig.model_validate(_config(task="time_series", cv={"method": "time_series"}))
    assert cfg.task is Task.TIME_SERIES


def test_group_cv_requires_group_col() -> None:
    with pytest.raises(ValidationError, match="group_col"):
        ExperimentConfig.model_validate(_config(cv={"method": "group"}))


def test_stratified_cv_rejects_regression() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_config(cv={"method": "stratified"}))


def test_time_cutoff_requires_time_column() -> None:
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(
            _config(task="time_series", cv={"method": "time_cutoff", "cutoffs": ["2024-01-01"]})
        )


def test_load_experiment_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "exp.yaml"
    path.write_text(
        """
name: yaml_exp
task: binary
data: {train_path: data/train.csv, target: 目的変数}
features:
  - {class: feature_engineering.numeric.LogTransformer, params: {variables: [x]}}
model: {name: xgboost, params: {n_estimators: 10}}
metrics: [auc, logloss]
""",
        encoding="utf-8",
    )
    cfg = load_experiment_config(path)
    assert cfg.data.target == "目的変数"
    assert cfg.features[0].class_path == "feature_engineering.numeric.LogTransformer"
    assert cfg.model.params == {"n_estimators": 10}


# --- tasks --------------------------------------------------------------------


def test_encode_target_classification_maps_to_integers() -> None:
    y, classes = encode_target(np.array(["b", "a", "b"]), Task.BINARY)
    assert y.tolist() == [1, 0, 1]
    assert classes is not None and classes.tolist() == ["a", "b"]


def test_encode_target_binary_requires_two_classes() -> None:
    with pytest.raises(ValueError):
        encode_target(np.array([0, 1, 2]), Task.BINARY)


def test_encode_target_rejects_missing_regression_target() -> None:
    with pytest.raises(ValueError):
        encode_target(np.array([1.0, np.nan]), Task.REGRESSION)


class _FakeClassifier:
    """一部のクラスしか学習していない分類器の代わり。"""

    classes_ = np.array([0, 2])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.tile(np.array([0.25, 0.75], dtype=np.float32), (len(X), 1))


def test_predict_multiclass_realigns_missing_classes() -> None:
    pred = predict(_FakeClassifier(), np.zeros((2, 1)), Task.MULTICLASS, n_classes=3)
    assert pred.shape == (2, 3)
    assert pred[0].tolist() == pytest.approx([0.25, 0.0, 0.75])
    assert pred.sum(axis=1) == pytest.approx([1.0, 1.0])


def test_predict_binary_without_positive_class_returns_zero() -> None:
    class OnlyNegative:
        classes_ = np.array([0])

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            return np.ones((len(X), 1))

    assert predict(OnlyNegative(), np.zeros((3, 1)), Task.BINARY).tolist() == [0.0, 0.0, 0.0]
