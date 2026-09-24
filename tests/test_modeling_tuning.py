"""modeling.tuning のテスト。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import optuna
import polars as pl
import pytest
import yaml

from modeling.config import ExperimentConfig
from modeling.experiment import Dataset, prepare_dataset, run_experiment
from modeling.tracking import MLflowTracker
from modeling.tuning import suggest_from_space, suggest_params, tune


def _config(model: str = "lightgbm", **tuning: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": f"tune_{model}",
            "task": "regression",
            "data": {
                "train_path": "unused.csv",
                "target": "y_reg",
                "drop_cols": ["id", "cat", "g", "ts", "y_bin", "y_multi"],
            },
            "cv": {"method": "kfold", "n_splits": 3},
            "model": {"name": model, "params": {"n_estimators": 20} if model != "linear" else {}},
            "metrics": ["rmse"],
            "tuning": tuning,
        }
    )


@pytest.fixture
def dataset(synthetic_frame: pl.DataFrame) -> Dataset:
    return prepare_dataset(_config(), synthetic_frame)


def test_suggest_from_space_supports_all_types() -> None:
    space = {
        "lr": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
        "leaves": {"type": "int", "low": 4, "high": 8},
        "boosting": {"type": "categorical", "choices": ["gbdt", "dart"]},
        "freq": {"type": "fixed", "value": 1},
    }
    trial = optuna.trial.FixedTrial({"lr": 0.05, "leaves": 6, "boosting": "dart"})
    assert suggest_from_space(trial, space) == {
        "lr": 0.05,
        "leaves": 6,
        "boosting": "dart",
        "freq": 1,
    }


def test_suggest_from_space_rejects_unknown_type() -> None:
    with pytest.raises(ValueError):
        suggest_from_space(optuna.trial.FixedTrial({}), {"x": {"type": "weird"}})


def test_suggest_params_overrides_base_params() -> None:
    cfg = _config(search_space={"learning_rate": {"type": "float", "low": 0.01, "high": 0.2}})
    params = suggest_params(optuna.trial.FixedTrial({"learning_rate": 0.1}), cfg)
    # 設定のparams（n_estimators）は残り、探索対象だけが上書きされる
    assert params == {"n_estimators": 20, "learning_rate": 0.1}


def test_tune_default_search_space_returns_complete_best_params(dataset: Dataset) -> None:
    result = tune(_config(), dataset, n_trials=3)
    assert result.n_trials == 3
    assert result.trials.height == 3
    # 既定探索空間の固定値（subsample_freq）と設定のn_estimatorsも含まれる
    assert result.best_params["n_estimators"] == 20
    assert result.best_params["subsample_freq"] == 1
    assert 1e-3 <= result.best_params["learning_rate"] <= 0.3
    assert result.best_value == pytest.approx(result.trials["value"].min())


def test_tune_linear_with_custom_space(dataset: Dataset) -> None:
    cfg = _config(
        "linear", search_space={"alpha": {"type": "float", "low": 0.01, "high": 10, "log": True}}
    )
    result = tune(cfg, dataset, n_trials=2)
    assert set(result.best_params) == {"alpha"}


def test_tune_resumes_study_from_storage(dataset: Dataset, tmp_path: Path) -> None:
    storage = tmp_path / "study.db"
    tune(_config(), dataset, n_trials=2, storage_path=storage)
    resumed = tune(_config(), dataset, n_trials=2, storage_path=storage)
    assert resumed.n_trials == 4


def test_tune_direction_follows_metric(synthetic_frame: pl.DataFrame) -> None:
    cfg = ExperimentConfig.model_validate(
        {
            "name": "tune_auc",
            "task": "binary",
            "data": {
                "train_path": "unused.csv",
                "target": "y_bin",
                "drop_cols": ["id", "cat", "g", "ts", "y_reg", "y_multi"],
            },
            "cv": {"method": "stratified", "n_splits": 3},
            "model": {"name": "lightgbm", "params": {"n_estimators": 10}},
            "metrics": ["auc"],
        }
    )
    ds = prepare_dataset(cfg, synthetic_frame)
    result = tune(cfg, ds, n_trials=3)
    # AUCは最大化なので最良値は試行中の最大値
    assert result.best_value == pytest.approx(result.trials["value"].max())


def test_run_experiment_with_tuning_logs_child_runs(dataset: Dataset, tmp_path: Path) -> None:
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = MLflowTracker("tune_exp", tracking_uri=uri, artifact_root=tmp_path / "artifacts")
    result = run_experiment(
        _config(),
        dataset,
        tracker=tracker,
        output_root=tmp_path / "out",
        tune_params=True,
        n_trials=2,
        optuna_dir=tmp_path / "optuna",
    )
    assert result.tuning_result is not None
    assert result.cv_result.params == result.tuning_result.best_params
    assert (result.output_dir / "tuning_trials.csv").is_file()
    assert (tmp_path / "optuna" / "tune_lightgbm.db").is_file()
    children = mlflow.search_runs(
        experiment_names=["tune_exp"],
        filter_string=f"tags.mlflow.parentRunId = '{result.run_id}'",
    )
    assert len(children) == 2


def test_run_experiment_script_with_tune(synthetic_frame: pl.DataFrame, tmp_path: Path) -> None:
    import run_experiment as script

    train_path = tmp_path / "train.parquet"
    synthetic_frame.write_parquet(train_path)
    raw = _config().model_dump(mode="json", by_alias=True)
    raw["data"]["train_path"] = str(train_path)
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    script.main(
        [
            "--config",
            str(config_path),
            "--no-tracking",
            "--tune",
            "--n-trials",
            "2",
            "--optuna-dir",
            str(tmp_path / "optuna"),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    trials = pl.read_csv(next((tmp_path / "out").rglob("tuning_trials.csv")))
    assert trials.height == 2
