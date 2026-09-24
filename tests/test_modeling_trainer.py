"""modeling.pipeline / modeling.trainer / modeling.models のテスト。"""

from __future__ import annotations

from typing import Any, ClassVar, Self

import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, TransformerMixin

from modeling.config import ExperimentConfig
from modeling.experiment import prepare_dataset
from modeling.models import available_models, get_model_spec
from modeling.pipeline import DropColumns, ToModelInput, build_pipeline, import_class
from modeling.trainer import run_cv

_TASK_SETTINGS = {
    "regression": ("y_reg", ["rmse", "mae"]),
    "binary": ("y_bin", ["auc", "logloss"]),
    "multiclass": ("y_multi", ["logloss", "accuracy"]),
}
_TARGETS = ["y_reg", "y_bin", "y_multi"]


def _config(
    task: str = "regression", model_name: str = "lightgbm", **overrides: Any
) -> ExperimentConfig:
    target, metrics = _TASK_SETTINGS[task]
    raw: dict[str, Any] = {
        "name": f"{model_name}_{task}",
        "task": task,
        "data": {
            "train_path": "unused.csv",
            "target": target,
            "id_col": "id",
            "drop_cols": [t for t in _TARGETS if t != target] + ["ts", "g"],
        },
        "features": [
            {
                "class": "feature_engineering.categorical.PolarsOneHotEncoder",
                "params": {"variables": ["cat"]},
            },
            {"class": "modeling.pipeline.DropColumns", "params": {"columns": ["cat"]}},
        ],
        "cv": {"method": "kfold", "n_splits": 3},
        "model": {"name": model_name, "params": _small_params(model_name)},
        "metrics": metrics,
    }
    return ExperimentConfig.model_validate(raw | overrides)


def _small_params(model: str) -> dict[str, Any]:
    # テストを高速にするため木の本数を減らす
    return {"n_estimators": 30} if model in ("lightgbm", "xgboost") else {}


# --- pipeline -----------------------------------------------------------------


def test_to_model_input_rejects_non_numeric_columns() -> None:
    frame = pl.DataFrame({"a": [1.0, 2.0], "s": ["x", "y"]})
    with pytest.raises(TypeError, match="s"):
        ToModelInput().fit(frame)


def test_to_model_input_keeps_fit_column_order_and_dtype() -> None:
    train = pl.DataFrame({"a": [1, 2], "b": [0.5, None]})
    t = ToModelInput(dtype="float32").fit(train)
    out = t.transform(train.select("b", "a"))
    assert list(out.columns) == ["a", "b"]
    assert str(out.dtypes["a"]) == "float32"
    assert np.isnan(out["b"].iloc[1])


def test_drop_columns() -> None:
    frame = pl.DataFrame({"a": [1], "b": [2]})
    assert DropColumns("a").fit_transform(frame).columns == ["b"]
    with pytest.raises(KeyError):
        DropColumns(["zzz"]).fit(frame)


def test_import_class_errors() -> None:
    assert import_class("modeling.pipeline.DropColumns") is DropColumns
    with pytest.raises(ImportError):
        import_class("DropColumns")
    with pytest.raises(ImportError):
        import_class("modeling.pipeline.NoSuchClass")


def test_build_pipeline_structure() -> None:
    pipeline = build_pipeline(_config())
    assert [name for name, _ in pipeline.steps][-2:] == ["to_model_input", "model"]
    assert pipeline.steps[0][0] == "00_PolarsOneHotEncoder"


def test_model_registry_contains_builtin_models() -> None:
    assert {"dummy", "linear", "lightgbm", "xgboost"} <= set(available_models())
    with pytest.raises(KeyError):
        get_model_spec("no_such_model")


# --- trainer ------------------------------------------------------------------


@pytest.mark.parametrize("model", ["dummy", "linear", "lightgbm", "xgboost"])
@pytest.mark.parametrize("task", ["regression", "binary", "multiclass"])
def test_run_cv_all_models_share_same_interface(
    synthetic_frame: pl.DataFrame, model: str, task: str
) -> None:
    cfg = _config(task, model)
    ds = prepare_dataset(cfg, synthetic_frame, synthetic_frame.head(7))
    result = run_cv(cfg, ds.X, ds.y, ds.folds, X_test=ds.X_test, n_classes=ds.n_classes)
    n = synthetic_frame.height
    expected_shape = (n, 3) if task == "multiclass" else (n,)
    assert result.oof_pred.shape == expected_shape
    # KFoldでは全行がちょうど1回ずつ予測される
    assert not np.isnan(result.oof_pred).any()
    assert (result.fold_ids >= 0).all()
    assert result.test_pred is not None
    assert result.test_pred.shape[0] == 7
    assert set(result.fold_scores) == set(cfg.metrics)
    assert all(len(v) == 3 for v in result.fold_scores.values())


def test_run_cv_learns_signal(synthetic_frame: pl.DataFrame) -> None:
    cfg = _config("regression", "linear")
    ds = prepare_dataset(cfg, synthetic_frame)
    model = run_cv(cfg, ds.X, ds.y, ds.folds)
    baseline_cfg = _config("regression", "dummy")
    baseline = run_cv(baseline_cfg, ds.X, ds.y, ds.folds)
    assert model.oof_scores["rmse"] < baseline.oof_scores["rmse"] / 5


class RecordingTransformer(BaseEstimator, TransformerMixin):
    """fitで受け取った行（idの値）を記録するtransformer（リーク検証用）。"""

    seen_in_fit: ClassVar[list[set[int]]] = []

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        RecordingTransformer.seen_in_fit.append(set(X["a_id"].to_list()))
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        return X


def test_run_cv_fits_preprocessing_on_train_fold_only(synthetic_frame: pl.DataFrame) -> None:
    RecordingTransformer.seen_in_fit.clear()
    frame = synthetic_frame.with_columns(a_id=pl.col("id"))
    cfg = _config(
        "regression",
        "linear",
        features=[{"class": "test_modeling_trainer.RecordingTransformer"}],
    )
    cfg = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"drop_cols": [*cfg.data.drop_cols, "cat"]})}
    )
    ds = prepare_dataset(cfg, frame)
    run_cv(cfg, ds.X, ds.y, ds.folds)
    assert len(RecordingTransformer.seen_in_fit) == len(ds.folds)
    for seen, (train_idx, valid_idx) in zip(
        RecordingTransformer.seen_in_fit, ds.folds, strict=True
    ):
        ids = frame["id"].to_numpy()
        assert seen == set(ids[train_idx].tolist())
        assert seen.isdisjoint(ids[valid_idx].tolist())


@pytest.mark.parametrize("model", ["lightgbm", "xgboost"])
def test_early_stopping_records_best_iteration(synthetic_frame: pl.DataFrame, model: str) -> None:
    cfg = _config(
        "regression",
        model,
        model={"name": model, "params": {"n_estimators": 500}, "early_stopping_rounds": 5},
    )
    ds = prepare_dataset(cfg, synthetic_frame)
    result = run_cv(cfg, ds.X, ds.y, ds.folds)
    assert all(b is not None and 1 <= b <= 500 for b in result.best_iterations)


def test_refit_full_trains_additional_model_with_mean_iterations(
    synthetic_frame: pl.DataFrame,
) -> None:
    cfg = _config(
        "regression",
        "lightgbm",
        model={"name": "lightgbm", "params": {"n_estimators": 500}, "early_stopping_rounds": 5},
        test_prediction="refit_full",
    )
    ds = prepare_dataset(cfg, synthetic_frame, synthetic_frame.head(5))
    result = run_cv(cfg, ds.X, ds.y, ds.folds, X_test=ds.X_test)
    assert len(result.models) == len(ds.folds) + 1
    iters = [b for b in result.best_iterations if b is not None]
    assert result.models[-1].named_steps["model"].n_estimators == int(np.mean(iters))
    assert result.test_pred is not None and result.test_pred.shape == (5,)


def test_time_series_cv_leaves_initial_rows_unpredicted(synthetic_frame: pl.DataFrame) -> None:
    raw = _config("regression", "linear").model_dump(mode="json", by_alias=True)
    raw.update(task="time_series", cv={"method": "time_series"})
    raw["data"]["time_col"] = "ts"
    cfg = ExperimentConfig.model_validate(raw)
    # 時刻の降順に並べたデータを渡しても、prepare_datasetで時刻順にソートされる
    ds = prepare_dataset(cfg, synthetic_frame.reverse())
    assert ds.X.height == synthetic_frame.height
    result = run_cv(cfg, ds.X, ds.y, ds.folds)
    first_valid = ds.folds[0][1].min()
    assert np.isnan(result.oof_pred[:first_valid]).all()
    assert not np.isnan(result.oof_pred[first_valid:]).any()
    assert np.isfinite(result.oof_scores["rmse"])
