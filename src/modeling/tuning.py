"""Optunaによるハイパーパラメータチューニング。

目的関数は「`run_cv` で得た主指標のfold平均」。CV分割は `Dataset.folds` を全試行で
使い回すため、試行間のスコア差は純粋にパラメータの差になる。fold毎のスコアを
`trial.report` に渡し、見込みの薄い試行はMedianPrunerで途中打ち切りする。
studyはSQLiteに保存するため、同じ実験名で再実行すると続きから探索できる。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import polars as pl

from modeling.config import ExperimentConfig
from modeling.metrics import get_metric
from modeling.models import get_model_spec
from modeling.tracking import NullTracker, Tracker
from modeling.trainer import run_cv

if TYPE_CHECKING:  # 型チェック時のみ参照（experimentとの循環importを避ける）
    from modeling.experiment import Dataset

# 試行ごとのINFOログを抑制する（結果は TuningResult.trials とMLflowで確認する）
optuna.logging.set_verbosity(optuna.logging.WARNING)
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")


@dataclass
class TuningResult:
    """チューニング結果。

    Attributes:
        best_params: 最良試行のモデルパラメータ（固定パラメータも含む完全な形）。
        best_value: 最良試行の主指標（fold平均）。
        n_trials: study内の完了済み試行数（再開前の試行も含む）。
        trials: 試行一覧（番号・状態・スコア・パラメータ）。
    """

    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    trials: pl.DataFrame


def suggest_from_space(trial: optuna.Trial, search_space: dict[str, dict[str, Any]]) -> dict:
    """YAMLで書いた探索空間からパラメータをサンプリングする。

    探索空間の書式（パラメータ名ごと）:
        - `{type: float, low: 1e-3, high: 0.3, log: true}`
        - `{type: int, low: 8, high: 256, log: false, step: 1}`
        - `{type: categorical, choices: [gbdt, dart]}`
        - `{type: fixed, value: 1}`（探索せず固定）

    Raises:
        ValueError: 未知の `type` の場合。
    """
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        kind = spec.get("type")
        if kind == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False), step=spec.get("step")
            )
        elif kind == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False),
                step=spec.get("step", 1),
            )
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        elif kind == "fixed":
            params[name] = spec["value"]
        else:
            raise ValueError(f"探索空間 {name} の type が不正です: {kind}")
    return params


def suggest_params(trial: optuna.Trial, config: ExperimentConfig) -> dict[str, Any]:
    """1試行分のモデルパラメータを作る（設定のparamsに、サンプリング結果を上書き）。"""
    if config.tuning.search_space is not None:
        sampled = suggest_from_space(trial, config.tuning.search_space)
    else:
        sampled = get_model_spec(config.model.name).search_space(trial, config.task)
    return {**config.model.params, **sampled}


def make_study(config: ExperimentConfig, storage_path: Path | None = None) -> optuna.Study:
    """studyを作る（`storage_path` のSQLiteに同名studyがあれば読み込んで再開）。"""
    metric = get_metric(config.primary_metric)
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        if config.tuning.pruning
        else optuna.pruners.NopPruner()
    )
    storage = None
    if storage_path is not None:
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_path.as_posix()}"
    return optuna.create_study(
        study_name=config.name,
        storage=storage,
        load_if_exists=True,
        direction=metric.direction,
        sampler=optuna.samplers.TPESampler(seed=config.seed),
        pruner=pruner,
    )


def tune(
    config: ExperimentConfig,
    dataset: Dataset,
    n_trials: int | None = None,
    timeout: float | None = None,
    storage_path: Path | None = None,
    tracker: Tracker | None = None,
) -> TuningResult:
    """ハイパーパラメータを探索する。

    Args:
        config: 実験設定。
        dataset: 学習用データ一式（CV分割を含む）。
        n_trials: 今回追加で実行する試行数（Noneなら `config.tuning.n_trials`）。
        timeout: 打ち切り秒数（Noneなら `config.tuning.timeout`）。
        storage_path: studyを保存するSQLiteファイル（Noneならメモリ上のみ）。
        tracker: 各試行を子runとして記録する記録先（親runが開始済みであること）。

    Returns:
        チューニング結果。
    """
    tracker = tracker or NullTracker()
    study = make_study(config, storage_path)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, config)
        # 固定パラメータも含めた完全な形で保存し、最良パラメータの復元に使う
        trial.set_user_attr("params", params)

        def report(fold: int, score: float) -> None:
            trial.report(score, step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        result = run_cv(
            config,
            dataset.X,
            dataset.y,
            dataset.folds,
            params=params,
            n_classes=dataset.n_classes,
            fold_callback=report,
        )
        value = result.mean_scores()[config.primary_metric]
        iters = [b for b in result.best_iterations if b is not None]
        if iters:
            trial.set_user_attr("mean_best_iteration", float(np.mean(iters)))
        with tracker.start_run(run_name=f"trial_{trial.number}", nested=True):
            tracker.log_params(params)
            tracker.log_metrics({f"cv_mean_{config.primary_metric}": value})
        return value

    study.optimize(
        objective,
        n_trials=n_trials if n_trials is not None else config.tuning.n_trials,
        timeout=timeout if timeout is not None else config.tuning.timeout,
        catch=(),
    )
    completed = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    if not completed:
        raise RuntimeError("完了した試行がありません（全試行が打ち切られた可能性があります）")
    best = study.best_trial
    return TuningResult(
        best_params=dict(best.user_attrs["params"]),
        best_value=float(study.best_value),
        n_trials=len(completed),
        trials=trials_frame(study),
    )


def trials_frame(study: optuna.Study) -> pl.DataFrame:
    """試行一覧をDataFrameにする（パラメータは `param_` 接頭辞の列）。"""
    rows = [
        {
            "number": t.number,
            "state": t.state.name,
            "value": t.value,
            **{f"param_{k}": v for k, v in t.params.items()},
        }
        for t in study.trials
    ]
    return pl.DataFrame(rows)
