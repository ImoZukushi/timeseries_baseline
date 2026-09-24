"""modeling.experiment / modeling.tracking / modeling.io / scripts/run_experiment.py のテスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from modeling.config import ExperimentConfig
from modeling.experiment import prepare_dataset, run_experiment, select_feature_columns
from modeling.io import (
    OOF_FILENAME,
    TEST_FILENAME,
    frame_to_predictions,
    prediction_columns,
    predictions_to_frame,
)
from modeling.tracking import MLflowTracker, flatten_dict


def _raw_config(train_path: Path, **overrides: object) -> dict[str, object]:
    raw: dict[str, object] = {
        "name": "exp_test",
        "task": "regression",
        "data": {
            "train_path": str(train_path),
            "target": "y_reg",
            "id_col": "id",
            "drop_cols": ["y_bin", "y_multi", "ts", "cat"],
        },
        "cv": {"method": "kfold", "n_splits": 3},
        "model": {"name": "lightgbm", "params": {"n_estimators": 20}},
        "metrics": ["rmse", "mae"],
    }
    return raw | overrides


# --- io -------------------------------------------------------------------------


def test_predictions_frame_round_trip_binary_and_multiclass() -> None:
    pred = np.array([0.1, np.nan, 0.9])
    frame = predictions_to_frame(pred, fold_ids=np.array([0, -1, 1]), target=np.array([0, 1, 1]))
    assert prediction_columns(frame) == ["pred"]
    assert frame["pred"].null_count() == 1  # NaNはnullとして保存
    back = frame_to_predictions(frame)
    assert np.isnan(back[1]) and back[[0, 2]].tolist() == [0.1, 0.9]

    multi = np.array([[0.2, 0.8], [0.6, 0.4]])
    frame_m = predictions_to_frame(multi, ids=pl.Series(["a", "b"]))
    assert prediction_columns(frame_m) == ["pred_0", "pred_1"]
    assert frame_m["id"].to_list() == ["a", "b"]
    assert np.allclose(frame_to_predictions(frame_m), multi)


# --- experiment -------------------------------------------------------------------


def test_select_feature_columns_default_and_explicit(tmp_path: Path) -> None:
    cfg = ExperimentConfig.model_validate(_raw_config(tmp_path / "t.csv"))
    cols = ["id", "a", "b", "g", "cat", "ts", "y_reg", "y_bin", "y_multi"]
    assert select_feature_columns(cfg, cols) == ["a", "b", "g"]
    raw = _raw_config(tmp_path / "t.csv")
    raw["data"]["feature_cols"] = ["a"]  # type: ignore[index]
    assert select_feature_columns(ExperimentConfig.model_validate(raw), cols) == ["a"]


def test_prepare_dataset_missing_column_raises(
    synthetic_frame: pl.DataFrame, tmp_path: Path
) -> None:
    raw = _raw_config(tmp_path / "t.csv")
    raw["data"]["target"] = "no_such"  # type: ignore[index]
    with pytest.raises(KeyError):
        prepare_dataset(ExperimentConfig.model_validate(raw), synthetic_frame)


def test_run_experiment_saves_outputs(synthetic_frame: pl.DataFrame, tmp_path: Path) -> None:
    cfg = ExperimentConfig.model_validate(_raw_config(tmp_path / "t.csv"))
    ds = prepare_dataset(cfg, synthetic_frame, synthetic_frame.head(4))
    result = run_experiment(cfg, ds, output_root=tmp_path / "out")
    out = result.output_dir
    assert out.parent == tmp_path / "out" / "exp_test"
    oof = pl.read_parquet(out / OOF_FILENAME)
    assert oof.columns == ["row", "id", "fold", "target", "pred"]
    assert oof.height == synthetic_frame.height
    test = pl.read_parquet(out / TEST_FILENAME)
    assert test.height == 4
    scores = pl.read_csv(out / "cv_scores.csv")
    assert scores["fold"].to_list() == ["0", "1", "2", "oof"]
    saved = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert saved["name"] == "exp_test"
    assert saved["resolved_params"] == {"n_estimators": 20}
    assert result.run_id is None


def test_run_experiment_logs_to_mlflow(synthetic_frame: pl.DataFrame, tmp_path: Path) -> None:
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = MLflowTracker("test_exp", tracking_uri=uri, artifact_root=tmp_path / "artifacts")
    cfg = ExperimentConfig.model_validate(_raw_config(tmp_path / "t.csv"))
    ds = prepare_dataset(cfg, synthetic_frame)
    result = run_experiment(cfg, ds, tracker=tracker, output_root=tmp_path / "out")
    assert result.run_id is not None
    run = mlflow.get_run(result.run_id)
    assert run.data.params["config.model.name"] == "lightgbm"
    assert run.data.params["resolved_params.n_estimators"] == "20"
    assert "cv_mean_rmse" in run.data.metrics
    assert "oof_mae" in run.data.metrics
    assert run.data.tags["model"] == "lightgbm"
    assert "git_commit" in run.data.tags
    artifacts = {p.name for p in (tmp_path / "artifacts").rglob("*") if p.is_file()}
    assert {"config.yaml", "cv_scores.csv", OOF_FILENAME} <= artifacts


def test_flatten_dict() -> None:
    assert flatten_dict({"a": {"b": 1, "c": {"d": 2}}, "e": 3}) == {"a.b": 1, "a.c.d": 2, "e": 3}


# --- CLI --------------------------------------------------------------------------


def test_run_experiment_script_end_to_end(
    synthetic_frame: pl.DataFrame, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import run_experiment as script

    train_path = tmp_path / "train.parquet"
    synthetic_frame.write_parquet(train_path)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    for model in ("dummy", "lightgbm"):
        raw = _raw_config(train_path, name=f"cli_{model}", model={"name": model})
        (config_dir / f"{model}.yaml").write_text(yaml.safe_dump(raw), encoding="utf-8")

    script.main(
        [
            "--config",
            str(config_dir / "*.yaml"),
            "--no-tracking",
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    printed = capsys.readouterr().out
    assert "cli_dummy" in printed and "cli_lightgbm" in printed
    assert (tmp_path / "out" / "cli_dummy").is_dir()
    assert (tmp_path / "out" / "cli_lightgbm").is_dir()


def test_run_experiment_script_missing_config_raises(tmp_path: Path) -> None:
    import run_experiment as script

    with pytest.raises(FileNotFoundError):
        script.main(["--config", str(tmp_path / "missing_*.yaml"), "--no-tracking"])
