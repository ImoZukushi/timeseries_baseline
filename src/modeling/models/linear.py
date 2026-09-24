"""ベースラインモデル（定数予測・線形モデル）。"""

from __future__ import annotations

from typing import Any

from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from modeling.models.base import ModelSpec, merge_params, register_model
from modeling.tasks import Task, is_classification

_ALL_TASKS = frozenset(Task)


@register_model
class DummySpec(ModelSpec):
    """定数予測（回帰は平均、分類は事前確率）。最低限超えるべきベースライン。"""

    name = "dummy"
    supported_tasks = _ALL_TASKS

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """定数予測モデルを作る。"""
        if is_classification(task):
            return DummyClassifier(**merge_params({"strategy": "prior"}, params))
        return DummyRegressor(**merge_params({"strategy": "mean"}, params))


@register_model
class LinearSpec(ModelSpec):
    """線形モデル（回帰はRidge、分類はLogisticRegression）。

    欠損値補完（中央値）と標準化を内部に含めたPipelineを返すため、
    前段の特徴量エンジニアリングで欠損を埋めておく必要はない。
    """

    name = "linear"
    supported_tasks = _ALL_TASKS

    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """欠損補完 + 標準化 + 線形モデルのPipelineを作る。"""
        model: BaseEstimator
        if is_classification(task):
            model = LogisticRegression(
                **merge_params({"max_iter": 1000, "random_state": seed}, params)
            )
        else:
            model = Ridge(**merge_params({"alpha": 1.0}, params))
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """正則化の強さを対数スケールで探索する。"""
        if is_classification(task):
            return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
        return {"alpha": trial.suggest_float("alpha", 1e-3, 1e3, log=True)}
