"""modeling.models.nn（skorch）のテスト。torch・skorch未インストール環境ではスキップする。"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import polars as pl
import pytest

pytest.importorskip("torch")
pytest.importorskip("skorch")

import torch  # noqa: E402

from modeling.config import ExperimentConfig  # noqa: E402
from modeling.experiment import prepare_dataset  # noqa: E402
from modeling.explain import compute_oof_shap  # noqa: E402
from modeling.models import available_models, get_model_spec  # noqa: E402
from modeling.models.nn import Conv1dModule, MLPModule  # noqa: E402
from modeling.tasks import Task  # noqa: E402
from modeling.trainer import run_cv  # noqa: E402

_TARGETS = {"regression": "y_reg", "binary": "y_bin", "multiclass": "y_multi"}
_METRICS = {"regression": ["rmse"], "binary": ["auc"], "multiclass": ["logloss"]}


def _config(task: str, model_name: str, **overrides: Any) -> ExperimentConfig:
    target = _TARGETS[task]
    raw: dict[str, Any] = {
        "name": f"{model_name}_{task}",
        "task": task,
        "data": {
            "train_path": "unused.csv",
            "target": target,
            "drop_cols": ["id", "cat", "g", "ts", *[t for t in _TARGETS.values() if t != target]],
        },
        "cv": {"method": "kfold", "n_splits": 3},
        "model": {"name": model_name, "params": {"max_epochs": 15}},
        "metrics": _METRICS[task],
    }
    return ExperimentConfig.model_validate(raw | overrides)


def test_nn_models_are_registered() -> None:
    assert {"mlp", "cnn1d"} <= set(available_models())


@pytest.mark.parametrize("kernel_size", [3, 4, 5])
def test_conv1d_module_output_shape(kernel_size: int) -> None:
    module = Conv1dModule(n_features=7, n_outputs=3, channels=(4, 8), kernel_size=kernel_size)
    assert module(torch.zeros(5, 7)).shape == (5, 3)


def test_mlp_module_output_shape() -> None:
    assert MLPModule(n_features=4, n_outputs=2, hidden_sizes=(8,))(torch.zeros(3, 4)).shape == (
        3,
        2,
    )


@pytest.mark.parametrize("model", ["mlp", "cnn1d"])
@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_nn_run_cv_same_interface(synthetic_frame: pl.DataFrame, task: str, model: str) -> None:
    cfg = _config(task, model)
    ds = prepare_dataset(cfg, synthetic_frame, synthetic_frame.head(5))
    result = run_cv(cfg, ds.X, ds.y, ds.folds, X_test=ds.X_test, n_classes=ds.n_classes)
    assert not np.isnan(result.oof_pred).any()
    assert result.test_pred is not None and result.test_pred.shape[0] == 5
    if task == "multiclass":
        assert result.oof_pred.sum(axis=1) == pytest.approx(np.ones(synthetic_frame.height))
    if task != "regression":
        assert ((result.oof_pred >= 0) & (result.oof_pred <= 1)).all()


def test_nn_learns_signal(synthetic_frame: pl.DataFrame) -> None:
    cfg = _config("regression", "mlp", model={"name": "mlp", "params": {"max_epochs": 60}})
    ds = prepare_dataset(cfg, synthetic_frame)
    model = run_cv(cfg, ds.X, ds.y, ds.folds)
    baseline = run_cv(_config("regression", "dummy", model={"name": "dummy"}), ds.X, ds.y, ds.folds)
    assert model.oof_scores["rmse"] < baseline.oof_scores["rmse"] / 2


def test_nn_is_reproducible_with_seed(synthetic_frame: pl.DataFrame) -> None:
    cfg = _config("regression", "mlp")
    ds = prepare_dataset(cfg, synthetic_frame)
    a = run_cv(cfg, ds.X, ds.y, ds.folds)
    b = run_cv(cfg, ds.X, ds.y, ds.folds)
    assert np.allclose(a.oof_pred, b.oof_pred)


def test_nn_early_stopping_and_refit(synthetic_frame: pl.DataFrame) -> None:
    cfg = _config(
        "regression",
        "mlp",
        model={"name": "mlp", "params": {"max_epochs": 200}, "early_stopping_rounds": 3},
        test_prediction="refit_full",
    )
    ds = prepare_dataset(cfg, synthetic_frame, synthetic_frame.head(5))
    result = run_cv(cfg, ds.X, ds.y, ds.folds, X_test=ds.X_test)
    iters = [b for b in result.best_iterations if b is not None]
    assert len(iters) == 3 and all(1 <= b <= 200 for b in iters)
    # 全データ再学習ではearly stoppingせず、fold平均の最良エポック数で学習する
    full_net = result.models[-1].named_steps["model"][-1]
    assert full_net.max_epochs == int(np.mean(iters))
    assert full_net.train_split is None


def test_nn_multiclass_with_missing_class_in_training() -> None:
    import pandas as pd

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"])
    y = np.where(X["a"].to_numpy() > 0, 2, 0)  # クラス1が学習データに無い
    model = get_model_spec("mlp").build(Task.MULTICLASS, {"max_epochs": 5}, seed=0).fit(X, y)
    assert model.predict_proba(X).shape == (60, 3)
    assert list(model.classes_) == [0, 1, 2]


def test_nn_permutation_shap(synthetic_frame: pl.DataFrame) -> None:
    cfg = _config("regression", "mlp", explain={"max_samples": 20})
    ds = prepare_dataset(cfg, synthetic_frame)
    result = run_cv(cfg, ds.X, ds.y, ds.folds)
    shap_result = compute_oof_shap(cfg, ds.X, ds.folds, result)
    assert shap_result.explainer_kind == "permutation"
    reconstructed = shap_result.values.sum(axis=1) + shap_result.base_values
    assert reconstructed == pytest.approx(result.oof_pred[shap_result.rows], abs=1e-4)


@pytest.mark.parametrize("model", ["mlp", "cnn1d"])
def test_nn_search_space(model: str) -> None:
    study = optuna.create_study()
    trial = study.ask()
    params = get_model_spec(model).search_space(trial, Task.REGRESSION)
    assert 1e-4 <= params["lr"] <= 1e-2
    assert "module__dropout" in params
