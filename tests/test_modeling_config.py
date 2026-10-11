"""modeling.config / modeling.tasks のテスト（指標は test_modeling_metrics.py）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from modeling.config import (
    ExperimentConfig,
    load_config_dict,
    load_ensemble_config,
    load_experiment_config,
)
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


# --- 設定の継承（base）と特徴量ブロック（use） -----------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
_BASE_YAML = """
name: base_exp
task: regression
data: {train_path: data/train.csv, target: y, drop_cols: [a, b]}
features:
  - {class: feature_engineering.numeric.LogTransformer, params: {variables: [x]}}
model: {name: lightgbm, params: {n_estimators: 100, learning_rate: 0.1}}
metrics: [rmse, mae]
forecast: null
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _block(path: Path, *classes: str) -> Path:
    """`steps` に指定クラスのステップを並べた特徴量ブロックのファイルを作る。"""
    steps = [{"class": c, "params": {"variables": ["x"]}} for c in classes]
    return _write(path, yaml.safe_dump({"description": "test", "steps": steps}))


def _step_names(cfg: ExperimentConfig) -> list[str]:
    return [s.class_path.rsplit(".", 1)[-1] for s in cfg.features]


def test_base_merges_dicts_and_replaces_lists(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", _BASE_YAML)
    child = _write(
        tmp_path / "child.yaml",
        """
base: base.yaml
name: child
data: {drop_cols: [a]}
model: {params: {learning_rate: 0.05}}
metrics: [mae]
""",
    )
    cfg = load_experiment_config(child)
    assert cfg.name == "child"
    # dict は再帰的にマージ（train_path・target は base のまま）、リストは置き換え
    assert cfg.data.train_path == Path("data/train.csv")
    assert cfg.data.drop_cols == ["a"]
    assert cfg.model.params == {"n_estimators": 100, "learning_rate": 0.05}
    assert cfg.metrics == ["mae"]
    assert _step_names(cfg) == ["LogTransformer"]
    assert "base" not in load_config_dict(child)


def test_base_null_override_multilevel_and_multiple(tmp_path: Path) -> None:
    _write(tmp_path / "configs" / "base.yaml", _BASE_YAML)
    _write(tmp_path / "configs" / "xgb.yaml", "model: {name: xgboost}\n")
    _write(tmp_path / "configs" / "mid.yaml", "base: base.yaml\nname: mid\nseed: 7\n")
    child = _write(
        tmp_path / "exp" / "child.yaml",
        "base: [../configs/mid.yaml, ../configs/xgb.yaml]\nname: child\nfeatures: null\n",
    )
    raw = load_config_dict(child)
    # 多段（mid → base）・複数（後ろが優先）・null での上書き
    assert raw["seed"] == 7
    assert raw["model"] == {
        "name": "xgboost",
        "params": {"n_estimators": 100, "learning_rate": 0.1},
    }
    assert raw["features"] is None
    assert raw["name"] == "child"


def test_base_errors(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "base: b.yaml\nname: a\n")
    _write(tmp_path / "b.yaml", "base: a.yaml\nname: b\n")
    with pytest.raises(ValueError, match="循環"):
        load_config_dict(tmp_path / "a.yaml")
    _write(tmp_path / "c.yaml", "base: missing.yaml\n")
    with pytest.raises(ValueError, match="見つかりません"):
        load_config_dict(tmp_path / "c.yaml")
    _write(tmp_path / "d.yaml", "base: {x: 1}\n")
    with pytest.raises(ValueError, match="base"):
        load_config_dict(tmp_path / "d.yaml")


def test_use_expands_blocks_by_name_and_path(tmp_path: Path) -> None:
    blocks = tmp_path / "features"
    _block(blocks / "log.yaml", "feature_engineering.numeric.LogTransformer")
    _block(tmp_path / "exp" / "local.yaml", "feature_engineering.numeric.SqrtTransformer")
    # ブロックの中の use（名前で参照）と通常のステップの混在
    _write(
        blocks / "nested.yaml",
        yaml.safe_dump(
            {
                "steps": [
                    {"use": "log"},
                    {
                        "class": "feature_engineering.numeric.ReciprocalTransformer",
                        "params": {"variables": ["x"]},
                    },
                ]
            }
        ),
    )
    exp = _write(
        tmp_path / "exp" / "exp.yaml",
        _BASE_YAML
        + """
features:
  - use: nested
  - {class: modeling.pipeline.DropColumns, params: {columns: [z]}}
  - use: local.yaml
""",
    )
    cfg = load_experiment_config(exp, feature_blocks_dir=blocks)
    assert _step_names(cfg) == [
        "LogTransformer",
        "ReciprocalTransformer",
        "DropColumns",
        "SqrtTransformer",
    ]


def test_use_in_base_is_resolved_relative_to_base_file(tmp_path: Path) -> None:
    _block(tmp_path / "common" / "blocks" / "b.yaml", "feature_engineering.numeric.SqrtTransformer")
    base = yaml.safe_load(_BASE_YAML) | {"features": [{"use": "blocks/b.yaml"}]}
    _write(tmp_path / "common" / "base.yaml", yaml.safe_dump(base, allow_unicode=True))
    child = _write(tmp_path / "exp" / "child.yaml", "base: ../common/base.yaml\nname: c\n")
    assert _step_names(load_experiment_config(child)) == ["SqrtTransformer"]


def test_use_errors(tmp_path: Path) -> None:
    blocks = tmp_path / "features"
    _write(blocks / "loop_a.yaml", "steps:\n  - use: loop_b\n")
    _write(blocks / "loop_b.yaml", "steps:\n  - use: loop_a\n")
    _write(blocks / "no_steps.yaml", "description: x\n")
    cases = {
        "- use: loop_a": "循環",
        "- use: missing": "見つかりません",
        "- {use: log, params: {a: 1}}": "他のキー",
        "- use: no_steps": "steps",
    }
    for i, (step, message) in enumerate(cases.items()):
        exp = _write(tmp_path / f"exp{i}.yaml", f"name: e\nfeatures:\n  {step}\n")
        with pytest.raises(ValueError, match=message):
            load_config_dict(exp, feature_blocks_dir=blocks)


def test_ensemble_config_supports_base(tmp_path: Path) -> None:
    _write(
        tmp_path / "base.yaml",
        "task: regression\nmethod: mean\nmetrics: [rmse]\n"
        "members:\n  - {name: a, experiment: a}\n  - {name: b, experiment: b}\n",
    )
    child = _write(tmp_path / "blend.yaml", "base: base.yaml\nname: blend\nmethod: weighted\n")
    cfg = load_ensemble_config(child)
    assert (cfg.name, cfg.method, len(cfg.members)) == ("blend", "weighted", 2)


def test_bundled_feature_search_configs_are_valid() -> None:
    paths = sorted((REPO_ROOT / "configs" / "experiments").glob("fe_*.yaml"))
    configs = {p.stem: load_experiment_config(p) for p in paths}
    assert {"fe_base", "fe_target_encoding", "fe_date_parts"} <= set(configs)
    assert _step_names(configs["fe_base"]) == ["PolarsOrdinalEncoder"]
    assert _step_names(configs["fe_target_encoding"]) == ["RareLabelGrouper", "PolarsTargetEncoder"]
    assert _step_names(configs["fe_date_parts"]) == [
        "PolarsOrdinalEncoder",
        "DatetimeFeaturesExtractor",
        "DropColumns",
    ]
    # 比較の前提: 全パターンでデータ・CV・モデルが同じ（data は drop_cols だけ違ってよい）
    base = configs["fe_base"]
    for cfg in configs.values():
        assert cfg.cv == base.cv and cfg.model == base.model and cfg.metrics == base.metrics
        assert cfg.data.train_path == base.data.train_path
        assert cfg.tracking.experiment_name == "feature_search"
    assert "年月日" not in configs["fe_date_parts"].data.drop_cols


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
