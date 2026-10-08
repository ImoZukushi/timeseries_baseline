"""modeling.evaluation（実験への誤差評価の組み込み）と、学習の推移の記録のテスト。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from modeling.config import EvaluationConfig, ExperimentConfig
from modeling.experiment import prepare_dataset, run_experiment
from modeling.models import get_model_spec
from modeling.pipeline import MODEL_STEP
from modeling.trainer import run_cv

_TARGETS = {"regression": "y_reg", "binary": "y_bin", "multiclass": "y_multi"}
_METRICS = {"regression": ["rmse"], "binary": ["auc"], "multiclass": ["logloss"]}

REGRESSION_FILES = {
    "residual_summary.csv",
    "residual_distribution.png",
    "residual_plot.png",
    "qq_plot.png",
    "leverage_cooks_distance.png",
    "top_cooks_distance.csv",
    "training_history.png",
}
CLASSIFICATION_FILES = {
    "confusion_matrix.png",
    "roc_curve.png",
    "pr_curve.png",
    "training_history.png",
}


def _config(task: str, model_name: str = "lightgbm", **overrides: Any) -> ExperimentConfig:
    target = _TARGETS[task]
    raw: dict[str, Any] = {
        "name": f"ev_{model_name}_{task}",
        "task": task,
        "data": {
            "train_path": "unused.csv",
            "target": target,
            "id_col": "id",
            "drop_cols": ["cat", "g", "ts", *[t for t in _TARGETS.values() if t != target]],
        },
        "cv": {"method": "kfold", "n_splits": 3},
        "model": {"name": model_name, "params": {"n_estimators": 60}, "early_stopping_rounds": 10},
        "metrics": _METRICS[task],
        "explain": {"enabled": False},
    }
    return ExperimentConfig.model_validate(raw | overrides)


def _run(frame: pl.DataFrame, config: ExperimentConfig, tmp_path: Path) -> Any:
    return run_experiment(config, prepare_dataset(config, frame), output_root=tmp_path)


def _files(result: Any) -> set[str]:
    directory = result.output_dir / "evaluation"
    return {p.name for p in directory.iterdir()} if directory.exists() else set()


# --- 設定 ------------------------------------------------------------------------------


def test_evaluation_config_validation() -> None:
    assert EvaluationConfig().enabled
    with pytest.raises(ValidationError, match="train_sizes"):
        EvaluationConfig.model_validate({"learning_curve": {"train_sizes": [0.0, 1.0]}})
    with pytest.raises(ValidationError, match="2つ以上"):
        EvaluationConfig.model_validate({"validation_curve": {"param": "alpha", "values": [1]}})


# --- タスクごとの出力 ---------------------------------------------------------------------


def test_regression_outputs(synthetic_frame: pl.DataFrame, tmp_path: Path) -> None:
    result = _run(synthetic_frame, _config("regression"), tmp_path)
    files = _files(result)
    assert files == REGRESSION_FILES  # 時刻列が無いので残差ACFは作らない
    top = pl.read_csv(result.output_dir / "evaluation" / "top_cooks_distance.csv")
    assert {"row", "leverage", "cooks_distance", "fold", "id"} <= set(top.columns)
    summary = pl.read_csv(result.output_dir / "evaluation" / "residual_summary.csv")
    # 残差の要約のRMSEは、実験のOOF RMSEと一致する
    assert summary["rmse"][0] == pytest.approx(result.cv_result.oof_scores["rmse"])


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_classification_outputs(synthetic_frame: pl.DataFrame, tmp_path: Path, task: str) -> None:
    assert _files(_run(synthetic_frame, _config(task), tmp_path)) == CLASSIFICATION_FILES


def test_disabled_evaluation_writes_nothing(synthetic_frame: pl.DataFrame, tmp_path: Path) -> None:
    result = _run(synthetic_frame, _config("regression", evaluation={"enabled": False}), tmp_path)
    assert not (result.output_dir / "evaluation").exists()


def test_linear_model_has_no_training_history(
    synthetic_frame: pl.DataFrame, tmp_path: Path
) -> None:
    config = _config("regression", model_name="linear", model={"name": "linear"})
    files = _files(_run(synthetic_frame, config, tmp_path))
    assert "training_history.png" not in files
    assert "residual_plot.png" in files


def test_learning_and_validation_curves_only_when_configured(
    synthetic_frame: pl.DataFrame, tmp_path: Path
) -> None:
    config = _config(
        "regression",
        evaluation={
            "learning_curve": {"enabled": True, "train_sizes": [0.5, 1.0]},
            "validation_curve": {"param": "learning_rate", "values": [0.05, 0.2]},
        },
    )
    result = _run(synthetic_frame, config, tmp_path)
    directory = result.output_dir / "evaluation"
    learning = pl.read_csv(directory / "learning_curve.csv")
    validation = pl.read_csv(directory / "validation_curve.csv")
    assert learning.height == 2
    assert validation["param_value"].cast(pl.Float64).to_list() == pytest.approx([0.05, 0.2])
    # スコアは元の向き（RMSEは正の値）で保存される
    assert (learning["valid_mean"] > 0).all()
    assert {"learning_curve.png", "validation_curve.png"} <= _files(result)


def _ts_frame(n_days: int = 200) -> pl.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for sid, level in (("A", 20.0), ("B", 50.0)):
        for i in range(n_days):
            rows.append(
                {
                    "date": dt.date(2024, 1, 1) + dt.timedelta(days=i),
                    "sid": sid,
                    "y": level + 3 * np.sin(2 * np.pi * i / 7) + rng.normal(),
                }
            )
    return pl.DataFrame(rows).sort("date", "sid")


def test_forecast_outputs_include_residual_correlogram(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "ev_ts",
            "task": "time_series",
            "data": {
                "train_path": "x.csv",
                "target": "y",
                "time_col": "date",
                "drop_cols": ["date"],
            },
            "features": [
                {
                    "class": "feature_engineering.categorical.PolarsOrdinalEncoder",
                    "params": {"variables": ["sid"]},
                }
            ],
            "cv": {"method": "time_cutoff", "cutoffs": ["2024-05-01", "2024-06-01"]},
            "model": {"name": "lightgbm", "params": {"n_estimators": 50}},
            "metrics": ["mae"],
            "forecast": {"series_col": "sid", "lags": [1, 7]},
            "explain": {"enabled": False},
            "evaluation": {"learning_curve": {"enabled": True, "train_sizes": [0.5, 1.0]}},
        }
    )
    result = _run(_ts_frame(), config, tmp_path)
    assert (REGRESSION_FILES | {"residual_acf_pacf.png", "learning_curve.png"}) <= _files(result)
    # ステップ別誤差の図は従来どおり実験の出力直下に保存される
    assert (result.output_dir / "horizon_error.png").is_file()


# --- 学習の推移 ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["lightgbm", "xgboost"])
def test_training_history_recorded_without_changing_early_stopping(
    synthetic_frame: pl.DataFrame, model: str
) -> None:
    with_history = _config(
        "regression",
        model_name=model,
        model={"name": model, "params": {"n_estimators": 300}, "early_stopping_rounds": 10},
    )
    without = with_history.model_copy(
        update={"evaluation": EvaluationConfig(training_history=False)}
    )
    ds = prepare_dataset(with_history, synthetic_frame)
    a = run_cv(with_history, ds.X, ds.y, ds.folds)
    b = run_cv(without, ds.X, ds.y, ds.folds)
    # 学習データを評価セットに加えても、early stopping の判定（最良反復）は変わらない
    assert a.best_iterations == b.best_iterations
    assert a.oof_pred == pytest.approx(b.oof_pred)
    spec = get_model_spec(model)
    for pipeline, best in zip(a.models, a.best_iterations, strict=True):
        history = spec.training_history(pipeline.named_steps[MODEL_STEP])
        assert history is not None and history.train is not None and history.valid is not None
        assert len(history.train) == len(history.valid)
        assert history.best_iteration == best
        assert len(history.valid) >= (best or 0)


def test_nn_training_history(synthetic_frame: pl.DataFrame) -> None:
    pytest.importorskip("skorch")
    config = _config(
        "regression",
        model_name="mlp",
        model={"name": "mlp", "params": {"max_epochs": 8}, "early_stopping_rounds": 3},
    )
    ds = prepare_dataset(config, synthetic_frame)
    result = run_cv(config, ds.X, ds.y, ds.folds)
    history = get_model_spec("mlp").training_history(result.models[0].named_steps[MODEL_STEP])
    assert history is not None and history.valid is not None
    assert len(history.train or []) == len(history.valid) <= 8


def _ts_config(**evaluation: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "ev_ts_series",
            "task": "time_series",
            "data": {
                "train_path": "x.csv",
                "target": "y",
                "time_col": "date",
                "drop_cols": ["date"],
            },
            "features": [
                {
                    "class": "feature_engineering.categorical.PolarsOrdinalEncoder",
                    "params": {"variables": ["sid"]},
                }
            ],
            "cv": {"method": "time_cutoff", "cutoffs": ["2024-05-01", "2024-06-01"]},
            "model": {"name": "lightgbm", "params": {"n_estimators": 30}},
            "metrics": ["mae"],
            "forecast": {"series_col": "sid", "lags": [1, 7]},
            "explain": {"enabled": False},
            "evaluation": evaluation,
        }
    )


def test_multi_series_residuals_are_split_per_series(tmp_path: Path) -> None:
    import matplotlib.image as mpimg

    result = _run(_ts_frame(), _ts_config(), tmp_path)
    directory = result.output_dir / "evaluation"
    summary = pl.read_csv(directory / "residual_summary.csv")
    # 全体の1行 ＋ 系列ごとの行
    assert summary["series"].to_list() == ["all", "A", "B"]
    assert summary.filter(pl.col("series") != "all")["n"].sum() == summary["n"][0]
    # 系列ごとに2枚のパネルを横に並べるため、1系列の図より横長になる
    single = _run(_ts_frame(), _ts_config(max_series=1), tmp_path / "single")
    wide = mpimg.imread(directory / "qq_plot.png").shape
    narrow = mpimg.imread(single.output_dir / "evaluation" / "qq_plot.png").shape
    assert wide[1] > narrow[1]
    # 図に描く系列を絞っても、要約表には全系列が残る
    single_summary = pl.read_csv(single.output_dir / "evaluation" / "residual_summary.csv")
    assert single_summary["series"].to_list() == ["all", "A", "B"]


def test_forecast_residual_diagnostics_recursive_and_onestep(tmp_path: Path) -> None:
    result = _run(_ts_frame(), _ts_config(ljung_box_lags=[3, 7]), tmp_path)
    files = _files(result)
    # 再帰予測と1期先予測の両方を、系列ごとに診断する
    assert {
        "residual_tests.csv",
        "ljung_box.csv",
        "residual_timeseries.png",
        "residual_timeseries_onestep.png",
        "residual_diagnostics__recursive__A.png",
        "residual_diagnostics__recursive__B.png",
        "residual_diagnostics__onestep__A.png",
        "residual_diagnostics__onestep__B.png",
    } <= files
    tests = pl.read_csv(result.output_dir / "evaluation" / "residual_tests.csv")
    assert tests.select("residual_type", "series").rows() == [
        ("recursive", "A"),
        ("recursive", "B"),
        ("onestep", "A"),
        ("onestep", "B"),
    ]
    lb = pl.read_csv(result.output_dir / "evaluation" / "ljung_box.csv")
    assert set(lb["lag"]) == {3, 7}


def test_forecast_residual_diagnostics_respect_max_series(tmp_path: Path) -> None:
    result = _run(_ts_frame(), _ts_config(max_series=1), tmp_path)
    files = _files(result)
    assert "residual_diagnostics__onestep__A.png" in files
    assert "residual_diagnostics__onestep__B.png" not in files
    # 検定の表は全系列
    tests = pl.read_csv(result.output_dir / "evaluation" / "residual_tests.csv")
    assert set(tests["series"]) == {"A", "B"}


def test_time_series_task_without_forecast_uses_oof_residuals(
    synthetic_frame: pl.DataFrame, tmp_path: Path
) -> None:
    raw = _config("regression").model_dump(mode="json", by_alias=True)
    raw.update(task="time_series", cv={"method": "time_series", "n_splits": 3})
    raw["data"]["time_col"] = "ts"
    result = _run(synthetic_frame, ExperimentConfig.model_validate(raw), tmp_path)
    files = _files(result)
    assert {
        "residual_tests.csv",
        "residual_timeseries.png",
        "residual_diagnostics__oof.png",
    } <= files
    tests = pl.read_csv(result.output_dir / "evaluation" / "residual_tests.csv")
    assert tests.select("residual_type", "series").rows() == [("oof", "all")]


def test_regression_without_time_col_has_no_time_series_diagnostics(
    synthetic_frame: pl.DataFrame, tmp_path: Path
) -> None:
    files = _files(_run(synthetic_frame, _config("regression"), tmp_path))
    assert not any(f.startswith(("residual_tests", "residual_diagnostics")) for f in files)


def test_single_series_summary_has_only_overall_row(
    synthetic_frame: pl.DataFrame, tmp_path: Path
) -> None:
    result = _run(synthetic_frame, _config("regression"), tmp_path)
    summary = pl.read_csv(result.output_dir / "evaluation" / "residual_summary.csv")
    assert summary["series"].to_list() == ["all"]
