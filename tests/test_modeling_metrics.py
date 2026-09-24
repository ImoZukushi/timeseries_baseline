"""modeling.metrics のテスト。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError
from sklearn import metrics as skm

from modeling.config import ExperimentConfig
from modeling.experiment import prepare_dataset, run_experiment
from modeling.metrics import available_metrics, get_metric
from modeling.tasks import Task
from modeling.tracking import MLflowTracker, sanitize_metric_key

# 二値: 閾値0.5のラベルは [0,1,1,0,1,0] → TP=2, FN=1, FP=1, TN=2
Y_BIN = np.array([0, 0, 1, 1, 1, 0])
P_BIN = np.array([0.1, 0.6, 0.7, 0.4, 0.9, 0.2])

# 多クラス: argmaxは [0,1,1,1,2,0]
Y_MULTI = np.array([0, 0, 1, 1, 2, 2])
P_MULTI = np.full((6, 3), 0.1)
P_MULTI[np.arange(6), [0, 1, 1, 1, 2, 0]] = 0.8


def _score(name: str, y: np.ndarray, p: np.ndarray) -> float:
    return get_metric(name)(y, p)


# --- 回帰 ------------------------------------------------------------------------


def test_regression_metrics() -> None:
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 5.0])
    assert _score("rmse", y, p) == pytest.approx(math.sqrt(4 / 3))
    assert _score("mae", y, p) == pytest.approx(2 / 3)
    assert _score("mape", np.array([1.0, 2.0, 4.0]), np.array([2.0, 2.0, 3.0])) == pytest.approx(
        (1.0 + 0.0 + 0.25) / 3
    )


def test_rmsle_clips_negative_predictions() -> None:
    y = np.array([0.0, 1.0])
    p = np.array([-1.0, math.e - 1])  # 負の予測は0として扱う
    assert _score("rmsle", y, p) == pytest.approx((1 - math.log(2)) / math.sqrt(2))


def test_rmsle_rejects_negative_target() -> None:
    with pytest.raises(ValueError, match="負の目的変数"):
        _score("rmsle", np.array([-1.0, 1.0]), np.array([1.0, 1.0]))


# --- 二値分類 -----------------------------------------------------------------------


def test_binary_label_metrics() -> None:
    assert _score("accuracy", Y_BIN, P_BIN) == pytest.approx(4 / 6)
    assert _score("precision", Y_BIN, P_BIN) == pytest.approx(2 / 3)
    assert _score("recall", Y_BIN, P_BIN) == pytest.approx(2 / 3)
    assert _score("f1", Y_BIN, P_BIN) == pytest.approx(2 / 3)
    # (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)) = (4-1)/9
    assert _score("mcc", Y_BIN, P_BIN) == pytest.approx(1 / 3)
    # sqrt(TPR * TNR) = sqrt(2/3 * 2/3)
    assert _score("g_mean", Y_BIN, P_BIN) == pytest.approx(2 / 3)
    assert _score("precision_micro", Y_BIN, P_BIN) == pytest.approx(4 / 6)
    assert _score("precision_macro", Y_BIN, P_BIN) == pytest.approx(2 / 3)


def test_binary_probability_metrics() -> None:
    # 陽性(0.7, 0.4, 0.9) と陰性(0.1, 0.6, 0.2) の全ペアのうち8/9で陽性が上
    assert _score("roc_auc", Y_BIN, P_BIN) == pytest.approx(8 / 9)
    # 降順で 1,1,0,1,0,0 → AP = 1/3*1 + 1/3*1 + 1/3*(3/4)
    assert _score("pr_auc", Y_BIN, P_BIN) == pytest.approx(11 / 12)
    assert _score("roc_auc_micro", Y_BIN, P_BIN) == pytest.approx(8 / 9)
    assert _score("roc_auc_macro", Y_BIN, P_BIN) == pytest.approx(8 / 9)


def test_auc_is_alias_of_roc_auc() -> None:
    assert _score("auc", Y_BIN, P_BIN) == _score("roc_auc", Y_BIN, P_BIN)
    # 別名で指定した場合も、名前は指定したまま（CV結果のキーが設定と一致するように）
    assert get_metric("auc").name == "auc"


def test_pauc_default_and_parameterized() -> None:
    expected = skm.roc_auc_score(Y_BIN, P_BIN, max_fpr=0.1)
    assert _score("pauc", Y_BIN, P_BIN) == pytest.approx(expected)
    assert _score("pauc@0.1", Y_BIN, P_BIN) == pytest.approx(expected)
    assert _score("pauc@0.5", Y_BIN, P_BIN) == pytest.approx(
        skm.roc_auc_score(Y_BIN, P_BIN, max_fpr=0.5)
    )
    # 上限1.0は通常のROC-AUCと同じ
    assert _score("pauc@1.0", Y_BIN, P_BIN) == pytest.approx(8 / 9)
    assert get_metric("pauc@0.05").name == "pauc@0.05"
    assert get_metric("pauc@0.05").tasks == frozenset({Task.BINARY})


@pytest.mark.parametrize("name", ["pauc@0", "pauc@1.5", "pauc@abc", "pauc@-0.1"])
def test_pauc_rejects_invalid_parameter(name: str) -> None:
    with pytest.raises(ValueError):
        get_metric(name)


def test_parameter_only_allowed_for_pauc() -> None:
    with pytest.raises(KeyError):
        get_metric("rmse@0.1")


def test_precision_without_predicted_positive_is_zero() -> None:
    # 陽性を1件も予測しない場合もエラー・警告にせず0とする
    assert _score("precision", np.array([0, 1]), np.array([0.1, 0.2])) == 0.0


# --- 多クラス分類 ---------------------------------------------------------------------


def test_multiclass_averaged_metrics() -> None:
    # クラス別 precision = (1/2, 2/3, 1), recall = (1/2, 1, 1/2), F1 = (1/2, 4/5, 2/3)
    assert _score("precision_macro", Y_MULTI, P_MULTI) == pytest.approx((1 / 2 + 2 / 3 + 1) / 3)
    assert _score("recall_macro", Y_MULTI, P_MULTI) == pytest.approx((1 / 2 + 1 + 1 / 2) / 3)
    assert _score("f1_macro", Y_MULTI, P_MULTI) == pytest.approx((1 / 2 + 4 / 5 + 2 / 3) / 3)
    # 多クラスのmicro平均は正解率と一致する
    for base in ("precision", "recall", "f1"):
        assert _score(f"{base}_micro", Y_MULTI, P_MULTI) == pytest.approx(4 / 6)
    # 各クラスの件数が同じなのでweightedはmacroと一致する
    for base in ("precision", "recall", "f1"):
        assert _score(f"{base}_weighted", Y_MULTI, P_MULTI) == pytest.approx(
            _score(f"{base}_macro", Y_MULTI, P_MULTI)
        )
    assert _score("g_mean", Y_MULTI, P_MULTI) == pytest.approx((1 / 2 * 1 * 1 / 2) ** (1 / 3))
    assert _score("mcc", Y_MULTI, P_MULTI) == pytest.approx(
        skm.matthews_corrcoef(Y_MULTI, [0, 1, 1, 1, 2, 0])
    )


def test_weighted_average_uses_class_support() -> None:
    y = np.array([0, 0, 0, 1])
    p = np.array([[0.9, 0.1], [0.9, 0.1], [0.1, 0.9], [0.1, 0.9]])  # ラベル 0,0,1,1
    # recall: クラス0=2/3（3件）、クラス1=1（1件）→ weighted = (3*2/3 + 1*1)/4
    assert _score("recall_weighted", y, p) == pytest.approx(3 / 4)
    assert _score("recall_macro", y, p) == pytest.approx((2 / 3 + 1) / 2)


def test_multiclass_roc_auc_micro_and_macro() -> None:
    onehot = np.eye(3)[Y_MULTI]
    assert _score("roc_auc_micro", Y_MULTI, P_MULTI) == pytest.approx(
        skm.roc_auc_score(onehot, P_MULTI, average="micro")
    )
    assert _score("roc_auc_macro", Y_MULTI, P_MULTI) == pytest.approx(
        skm.roc_auc_score(onehot, P_MULTI, average="macro")
    )


def test_multiclass_metrics_with_class_missing_from_fold() -> None:
    # 検証foldにクラス2が無くても、ラベル系・macro AUCはエラーにならない
    y = np.array([0, 0, 1, 1])
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.2, 0.7, 0.1]])
    for name in ("precision_macro", "recall_macro", "f1_weighted", "mcc", "roc_auc_micro"):
        assert np.isfinite(_score(name, y, p))
    # 予測にだけ現れるクラス2のRecallは定義上0なのでG-Meanも0
    assert _score("g_mean", y, p) == 0.0
    # macro AUCは存在するクラス（0, 1）のみの平均
    onehot = np.eye(3)[y][:, :2]
    assert _score("roc_auc_macro", y, p) == pytest.approx(
        skm.roc_auc_score(onehot, p[:, :2], average="macro")
    )


# --- レジストリ・設定との整合 ----------------------------------------------------------


def test_available_metrics_contains_requested_metrics() -> None:
    requested = {
        "mae", "mape", "rmsle", "accuracy", "mcc", "precision", "recall", "g_mean",
        "pr_auc", "roc_auc", "pauc",
        "precision_micro", "precision_macro", "precision_weighted",
        "recall_micro", "recall_macro", "recall_weighted",
        "f1_micro", "f1_macro", "f1_weighted",
        "roc_auc_micro", "roc_auc_macro",
    }  # fmt: skip
    assert requested <= set(available_metrics())
    with pytest.raises(KeyError):
        get_metric("nope")


@pytest.mark.parametrize("name", available_metrics())
def test_every_metric_is_computable_for_its_tasks(name: str) -> None:
    metric = get_metric(name)
    rng = np.random.default_rng(0)
    inputs: dict[Task, tuple[np.ndarray, np.ndarray]] = {
        Task.REGRESSION: (rng.uniform(1, 5, 50), rng.uniform(1, 5, 50)),
        Task.TIME_SERIES: (rng.uniform(1, 5, 50), rng.uniform(1, 5, 50)),
        Task.BINARY: (np.tile([0, 1], 25), rng.uniform(0, 1, 50)),
        Task.MULTICLASS: (np.tile([0, 1, 2], 20), rng.dirichlet(np.ones(3), 60)),
    }
    assert metric.tasks, f"{name} に対応タスクがありません"
    assert metric.direction in ("maximize", "minimize")
    for task in metric.tasks:
        y, p = inputs[task]
        assert np.isfinite(metric(y, p)), f"{name} / {task}"


def _config(task: str, metrics: list[str]) -> dict[str, Any]:
    return {
        "name": "m",
        "task": task,
        "data": {"train_path": "t.csv", "target": "y"},
        "model": {"name": "dummy"},
        "metrics": metrics,
    }


@pytest.mark.parametrize(
    ("task", "metric"),
    [
        ("multiclass", "pr_auc"),
        ("multiclass", "pauc"),
        ("multiclass", "precision"),
        ("regression", "pr_auc"),
        ("binary", "rmsle"),
        ("time_series", "roc_auc"),
    ],
)
def test_config_rejects_metric_for_unsupported_task(task: str, metric: str) -> None:
    raw = _config(task, [metric])
    if task == "time_series":
        raw["cv"] = {"method": "time_series"}
    with pytest.raises(ValidationError, match="対応していません"):
        ExperimentConfig.model_validate(raw)


def test_config_accepts_new_metrics() -> None:
    cfg = ExperimentConfig.model_validate(_config("binary", ["pr_auc", "pauc@0.05", "auc"]))
    assert cfg.primary_metric == "pr_auc"
    with pytest.raises(ValidationError):
        ExperimentConfig.model_validate(_config("binary", ["pauc@2"]))


def test_sanitize_metric_key() -> None:
    assert sanitize_metric_key("oof_pauc@0.05") == "oof_pauc_0.05"
    assert sanitize_metric_key("cv_mean_f1_macro") == "cv_mean_f1_macro"


def test_binary_experiment_with_new_metrics_logs_to_mlflow(
    synthetic_frame: pl.DataFrame, tmp_path: Path
) -> None:
    import mlflow

    metrics = ["pr_auc", "pauc@0.05", "mcc", "g_mean", "roc_auc_macro"]
    cfg = ExperimentConfig.model_validate(
        {
            "name": "binary_metrics",
            "task": "binary",
            "data": {
                "train_path": "unused.csv",
                "target": "y_bin",
                "drop_cols": ["id", "cat", "g", "ts", "y_reg", "y_multi"],
            },
            "cv": {"method": "stratified", "n_splits": 3},
            "model": {"name": "lightgbm", "params": {"n_estimators": 20}},
            "metrics": metrics,
            "explain": {"enabled": False},
        }
    )
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = MLflowTracker("metrics_exp", tracking_uri=uri, artifact_root=tmp_path / "art")
    result = run_experiment(
        cfg, prepare_dataset(cfg, synthetic_frame), tracker=tracker, output_root=tmp_path / "out"
    )
    # CV結果のキーは設定に書いた名前のまま
    assert set(result.cv_result.oof_scores) == set(metrics)
    logged = mlflow.get_run(result.run_id).data.metrics
    assert "oof_pauc_0.05" in logged
    assert logged["oof_pr_auc"] == pytest.approx(result.cv_result.oof_scores["pr_auc"])
    scores = pl.read_csv(result.output_dir / "cv_scores.csv")
    assert "pauc@0.05" in scores.columns
