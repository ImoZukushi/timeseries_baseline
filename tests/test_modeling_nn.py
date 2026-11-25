"""modeling.models.nn（skorch）のテスト。torch・skorch未インストール環境ではスキップする。"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("skorch")

import torch  # noqa: E402
from skorch.dataset import ValidSplit  # noqa: E402

from modeling.config import ExperimentConfig  # noqa: E402
from modeling.experiment import prepare_dataset  # noqa: E402
from modeling.explain import compute_oof_shap  # noqa: E402
from modeling.models import available_models, get_model_spec  # noqa: E402
from modeling.models.nn import (  # noqa: E402
    Conv1dModule,
    MLPModule,
    NetInputScaler,
    TabMModule,
)
from modeling.tasks import Task  # noqa: E402
from modeling.trainer import fit_pipeline, run_cv  # noqa: E402

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
    assert {"mlp", "cnn1d", "tabm"} <= set(available_models())


@pytest.mark.parametrize("kernel_size", [3, 4, 5])
def test_conv1d_module_output_shape(kernel_size: int) -> None:
    module = Conv1dModule(n_features=7, n_outputs=3, channels=(4, 8), kernel_size=kernel_size)
    assert module(torch.zeros(5, 7)).shape == (5, 3)


def test_mlp_module_output_shape() -> None:
    assert MLPModule(n_features=4, n_outputs=2, hidden_sizes=(8,))(torch.zeros(3, 4)).shape == (
        3,
        2,
    )


@pytest.mark.parametrize("model", ["mlp", "cnn1d", "tabm"])
@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_nn_run_cv_same_interface(synthetic_frame: pd.DataFrame, task: str, model: str) -> None:
    cfg = _config(task, model)
    ds = prepare_dataset(cfg, synthetic_frame, synthetic_frame.head(5))
    result = run_cv(cfg, ds.X, ds.y, ds.folds, X_test=ds.X_test, n_classes=ds.n_classes)
    assert not np.isnan(result.oof_pred).any()
    assert result.test_pred is not None and result.test_pred.shape[0] == 5
    if task == "multiclass":
        assert result.oof_pred.sum(axis=1) == pytest.approx(np.ones(len(synthetic_frame)))
    if task != "regression":
        assert ((result.oof_pred >= 0) & (result.oof_pred <= 1)).all()


def test_nn_learns_signal(synthetic_frame: pd.DataFrame) -> None:
    cfg = _config("regression", "mlp", model={"name": "mlp", "params": {"max_epochs": 60}})
    ds = prepare_dataset(cfg, synthetic_frame)
    model = run_cv(cfg, ds.X, ds.y, ds.folds)
    baseline = run_cv(_config("regression", "dummy", model={"name": "dummy"}), ds.X, ds.y, ds.folds)
    assert model.oof_scores["rmse"] < baseline.oof_scores["rmse"] / 2


def test_nn_is_reproducible_with_seed(synthetic_frame: pd.DataFrame) -> None:
    cfg = _config("regression", "mlp")
    ds = prepare_dataset(cfg, synthetic_frame)
    a = run_cv(cfg, ds.X, ds.y, ds.folds)
    b = run_cv(cfg, ds.X, ds.y, ds.folds)
    assert np.allclose(a.oof_pred, b.oof_pred)


def test_nn_early_stopping_and_refit(synthetic_frame: pd.DataFrame) -> None:
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


def test_nn_permutation_shap(synthetic_frame: pd.DataFrame) -> None:
    cfg = _config("regression", "mlp", explain={"max_samples": 20})
    ds = prepare_dataset(cfg, synthetic_frame)
    result = run_cv(cfg, ds.X, ds.y, ds.folds)
    shap_result = compute_oof_shap(cfg, ds.X, ds.folds, result)
    assert shap_result.explainer_kind == "permutation"
    reconstructed = shap_result.values.sum(axis=1) + shap_result.base_values
    assert reconstructed == pytest.approx(result.oof_pred[shap_result.rows], abs=1e-4)


@pytest.mark.parametrize("model", ["mlp", "cnn1d", "tabm"])
def test_nn_search_space(model: str) -> None:
    study = optuna.create_study()
    trial = study.ask()
    params = get_model_spec(model).search_space(trial, Task.REGRESSION)
    assert 1e-4 <= params["lr"] <= 1e-2
    assert "module__dropout" in params


@pytest.mark.parametrize("arch_type", ["tabm", "tabm-mini"])
@pytest.mark.parametrize("num_embeddings", ["periodic", None])
def test_tabm_module_output_shape(arch_type: str, num_embeddings: str | None) -> None:
    module = TabMModule(
        n_features=5, n_outputs=2, k=4, n_blocks=2, d_block=8, arch_type=arch_type,
        num_embeddings=num_embeddings, d_embedding=4, n_frequencies=3,
    )  # fmt: skip
    assert module(torch.zeros(3, 5)).shape == (3, 4, 2)


def test_tabm_predicts_member_mean() -> None:
    # 予測（評価モード）は k 個のメンバーの予測の平均
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])
    y = X["a"].to_numpy() * 3
    model = get_model_spec("tabm").build(Task.REGRESSION, {"max_epochs": 3}, seed=0).fit(X, y)
    net = model[-1]
    Xt = torch.as_tensor(model[:-1].transform(X), dtype=torch.float32)
    net.module_.eval()
    with torch.no_grad():
        members = net.module_(Xt).mean(dim=1).numpy().ravel()
    expected = members * net.y_scale_ + net.y_mean_
    assert model.predict(X) == pytest.approx(expected, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("model", ["mlp", "tabm"])
def test_nn_accepts_sample_weight(synthetic_frame: pd.DataFrame, model: str) -> None:
    cfg = _config("regression", model)
    ds = prepare_dataset(cfg, synthetic_frame)
    weight = np.where(np.arange(len(ds.y)) % 2 == 0, 3.0, 1.0)
    pipeline = fit_pipeline(cfg, ds.X, ds.y, sample_weight=weight)
    assert np.isfinite(pipeline.predict(ds.X)).all()


def test_nn_zero_weight_rows_are_ignored() -> None:
    # 重み 0 の行は損失に入らないので、その行の目的変数を変えても学習結果は変わらない
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(64, 2)), columns=["a", "b"])
    y = X["a"].to_numpy()
    weight = np.ones(64)
    weight[:8] = 0.0
    y_changed = y.copy()
    y_changed[:8] = 1000.0
    params = {"max_epochs": 5, "batch_size": 64, "standardize_target": False}
    spec = get_model_spec("mlp")
    a = spec.build(Task.REGRESSION, params, seed=0).fit(X, y, sample_weight=weight)
    b = spec.build(Task.REGRESSION, params, seed=0).fit(X, y_changed, sample_weight=weight)
    assert a.predict(X) == pytest.approx(b.predict(X), abs=1e-5)


@pytest.mark.parametrize("objective", ["l2", "l1", "huber", "quantile", "poisson"])
def test_nn_objectives(synthetic_frame: pd.DataFrame, objective: str) -> None:
    spec = get_model_spec("tabm")
    params = {"max_epochs": 5, **spec.objective_params(objective, None)}
    frame = synthetic_frame.assign(y_reg=synthetic_frame["y_reg"].abs())  # poisson は非負
    cfg = _config("regression", "tabm", model={"name": "tabm", "params": params})
    ds = prepare_dataset(cfg, frame)
    pred = fit_pipeline(cfg, ds.X, ds.y).predict(ds.X)
    assert np.isfinite(pred).all()
    if objective == "poisson":
        assert (pred > 0).all()


def test_nn_huber_threshold_is_in_original_scale() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(50, 2)), columns=["a", "b"])
    y = 100 + 10 * X["a"].to_numpy()
    params = {"max_epochs": 2, **get_model_spec("mlp").objective_params("huber", 5.0)}
    model = get_model_spec("mlp").build(Task.REGRESSION, params, seed=0).fit(X, y)
    net = model[-1]
    assert net.criterion_.alpha == pytest.approx(5.0 / np.std(y))


def test_nn_unknown_objective_raises() -> None:
    with pytest.raises(ValueError):
        get_model_spec("tabm").objective_params("unknown")


def test_nn_early_stopping_uses_validation_fold(synthetic_frame: pd.DataFrame) -> None:
    cfg = _config(
        "regression",
        "tabm",
        model={"name": "tabm", "params": {"max_epochs": 100}, "early_stopping_rounds": 3},
    )
    ds = prepare_dataset(cfg, synthetic_frame)
    train_idx, valid_idx = ds.folds[0]
    X_va = ds.X.iloc[valid_idx]
    pipeline = fit_pipeline(cfg, ds.X.iloc[train_idx], ds.y[train_idx], X_va, ds.y[valid_idx])
    net = pipeline.named_steps["model"][-1]
    # 学習データから切り出さず、渡した検証データ（全部の行）で検証の損失を計算している
    assert net.has_valid_ and net.early_stopping_
    batches = net.history[-1]["batches"]
    valid_rows = sum(b["valid_batch_size"] for b in batches if "valid_batch_size" in b)
    assert valid_rows == len(valid_idx)
    best = get_model_spec("tabm").best_iteration(pipeline.named_steps["model"])
    assert best is not None and 1 <= best <= len(net.history)
    # 学習の後は train_split を元（検証データを持たない ValidSplit）に戻す
    assert isinstance(net.train_split, ValidSplit)


def test_nn_history_only_has_no_best_iteration(synthetic_frame: pd.DataFrame) -> None:
    # 学習の推移を記録するだけ（early stopping なし）なら、最良エポック数は None（LightGBM と同じ）
    cfg = _config("regression", "mlp", evaluation={"enabled": True, "training_history": True})
    ds = prepare_dataset(cfg, synthetic_frame)
    train_idx, valid_idx = ds.folds[0]
    pipeline = fit_pipeline(
        cfg, ds.X.iloc[train_idx], ds.y[train_idx], ds.X.iloc[valid_idx], ds.y[valid_idx],
        early_stopping=False,
    )  # fmt: skip
    spec = get_model_spec("mlp")
    model = pipeline.named_steps["model"]
    assert spec.best_iteration(model) is None
    history = spec.training_history(model)
    assert history is not None and history.valid is not None
    assert len(history.valid) == len(history.train) == 15


def test_nn_n_estimators_alias() -> None:
    spec = get_model_spec("tabm")
    model = spec.build(Task.REGRESSION, {"n_estimators": 7}, seed=0)
    assert model[-1].max_epochs == 7
    assert spec.with_n_iterations({"n_estimators": 7, "lr": 0.1}, 3) == {"max_epochs": 3, "lr": 0.1}


def test_net_input_scaler_keeps_flags_standardized() -> None:
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.exponential(size=200), rng.integers(0, 2, 200)])
    out = NetInputScaler(method="quantile").fit_transform(X)
    # 連続の列は正規分布に写り、フラグの列は標準化される
    assert abs(np.median(out[:, 0])) < 0.1
    assert np.unique(out[:, 1]).size == 2
    assert out[:, 1].mean() == pytest.approx(0.0, abs=1e-9)


def test_tabm_search_space_is_bounded_for_cpu() -> None:
    # CPU で1試行が数分に収まるよう、計算量を決める値の上限を抑えている
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(200):
        params = get_model_spec("tabm").search_space(study.ask(), Task.REGRESSION)
        assert 1 <= params["module__n_blocks"] <= 3
        assert 64 <= params["module__d_block"] <= 256
        assert params["batch_size"] in (64, 128)
        assert params.get("module__d_embedding", 8) in (8, 16)
