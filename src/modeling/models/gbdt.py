"""勾配ブースティング決定木（LightGBM・XGBoost）。"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import xgboost as xgb
from sklearn.base import BaseEstimator

from modeling.models.base import ModelSpec, merge_params, register_model
from modeling.tasks import Task, is_classification

_ALL_TASKS = frozenset(Task)


@register_model
class LightGBMSpec(ModelSpec):
    """LightGBM（`LGBMRegressor` / `LGBMClassifier`）。"""

    name = "lightgbm"
    supported_tasks = _ALL_TASKS
    explainer_kind = "tree"

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """LightGBMのestimatorを作る（early stoppingは `fit_kwargs` のcallbackで指定）。"""
        merged = merge_params({"random_state": seed, "verbose": -1}, params)
        if is_classification(task):
            return lgb.LGBMClassifier(**merged)
        return lgb.LGBMRegressor(**merged)

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """LightGBMの代表的なパラメータの探索空間。"""
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 256, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def fit_kwargs(
        self, X_valid: Any, y_valid: Any, early_stopping_rounds: int | None
    ) -> dict[str, Any]:
        """early stopping用に検証データとcallbackを渡す。"""
        if early_stopping_rounds is None:
            return {}
        # LightGBM 4.7以降は eval_set ではなく eval_X / eval_y で渡す
        return {
            "eval_X": (X_valid,),
            "eval_y": (y_valid,),
            "callbacks": [lgb.early_stopping(early_stopping_rounds, verbose=False)],
        }

    def best_iteration(self, estimator: Any) -> int | None:
        """early stoppingで決まった最良イテレーション数。"""
        best = getattr(estimator, "best_iteration_", None)
        return int(best) if best else None

    def with_n_iterations(self, params: dict[str, Any], n_iterations: int) -> dict[str, Any]:
        """木の本数を固定したパラメータを返す。"""
        return {**params, "n_estimators": n_iterations}


@register_model
class XGBoostSpec(ModelSpec):
    """XGBoost（`XGBRegressor` / `XGBClassifier`）。"""

    name = "xgboost"
    supported_tasks = _ALL_TASKS
    explainer_kind = "tree"

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """XGBoostのestimatorを作る（XGBoost 2.0以降はearly stoppingをコンストラクタで指定）。"""
        defaults: dict[str, Any] = {"random_state": seed, "tree_method": "hist"}
        if early_stopping_rounds is not None:
            defaults["early_stopping_rounds"] = early_stopping_rounds
        merged = merge_params(defaults, params)
        if is_classification(task):
            return xgb.XGBClassifier(**merged)
        return xgb.XGBRegressor(**merged)

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """XGBoostの代表的なパラメータの探索空間。"""
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 100.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

    def fit_kwargs(
        self, X_valid: Any, y_valid: Any, early_stopping_rounds: int | None
    ) -> dict[str, Any]:
        """early stopping用に検証データを渡す。"""
        if early_stopping_rounds is None:
            return {}
        return {"eval_set": [(X_valid, y_valid)], "verbose": False}

    def best_iteration(self, estimator: Any) -> int | None:
        """early stoppingで決まった最良イテレーション数（0始まりのため+1する）。"""
        try:
            return int(estimator.best_iteration) + 1
        except AttributeError:
            # early stoppingなしで学習した場合は best_iteration が存在しない
            return None

    def with_n_iterations(self, params: dict[str, Any], n_iterations: int) -> dict[str, Any]:
        """木の本数を固定したパラメータを返す。"""
        return {**params, "n_estimators": n_iterations}
