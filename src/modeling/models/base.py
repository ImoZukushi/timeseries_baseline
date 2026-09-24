"""モデルの差異を吸収する `ModelSpec` と、そのレジストリ。

学習ループ・CV・チューニング・ログはモデル非依存に書き、モデル固有の知識
（estimatorの組み立て、探索空間、early stoppingの渡し方、SHAPの種類）は
すべて `ModelSpec` のサブクラスに閉じ込める。新しいモデルを追加するときは
`ModelSpec` を継承したクラスを1つ作り `@register_model` を付けるだけでよい。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal

from sklearn.base import BaseEstimator

from modeling.tasks import Task

ExplainerKind = Literal["tree", "linear", "permutation"]


class ModelSpec(ABC):
    """モデル1種類分の定義。

    Attributes:
        name: YAMLの `model.name` で指定する名前。
        supported_tasks: 対応するタスク。
        explainer_kind: SHAPで使うexplainerの種類。
        requires_float32: 入力特徴量をfloat32に変換する必要があるか（PyTorch系）。
    """

    name: ClassVar[str]
    supported_tasks: ClassVar[frozenset[Task]]
    explainer_kind: ClassVar[ExplainerKind] = "permutation"
    requires_float32: ClassVar[bool] = False

    @abstractmethod
    def build(
        self,
        task: Task,
        params: dict[str, Any],
        seed: int,
        early_stopping_rounds: int | None = None,
    ) -> BaseEstimator:
        """sklearn互換のestimatorを作る。

        Args:
            task: 予測タスク。
            params: YAMLまたはチューニングで与えられたハイパーパラメータ。
            seed: 乱数シード（`params` に明示されていればそちらを優先すること）。
            early_stopping_rounds: early stoppingのラウンド数（Noneなら無効）。

        Returns:
            未学習のestimator。
        """

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        """Optunaの既定探索空間からパラメータをサンプリングする。

        Args:
            trial: `optuna.Trial`。
            task: 予測タスク。

        Returns:
            サンプリングされたパラメータ。

        Raises:
            NotImplementedError: 既定探索空間を持たないモデルの場合。
        """
        raise NotImplementedError(f"{self.name} には既定の探索空間がありません")

    def fit_kwargs(
        self, X_valid: Any, y_valid: Any, early_stopping_rounds: int | None
    ) -> dict[str, Any]:
        """`estimator.fit` に渡す追加引数（early stopping用の検証データ等）を返す。

        既定では追加引数なし。
        """
        return {}

    def best_iteration(self, estimator: Any) -> int | None:
        """early stoppingで決まった最良イテレーション数を返す（該当しなければNone）。"""
        return None

    def with_n_iterations(self, params: dict[str, Any], n_iterations: int) -> dict[str, Any]:
        """全データ再学習時に、イテレーション数を固定したパラメータを返す。

        既定ではパラメータをそのまま返す（イテレーションの概念がないモデル）。
        """
        return params

    def supports(self, task: Task) -> bool:
        """タスクに対応しているかを返す。"""
        return task in self.supported_tasks


_REGISTRY: dict[str, ModelSpec] = {}


def register_model[SpecT: type[ModelSpec]](spec_cls: SpecT) -> SpecT:
    """`ModelSpec` サブクラスをレジストリに登録するデコレータ。

    Raises:
        ValueError: 同名のモデルが登録済みの場合。
    """
    name = spec_cls.name
    if name in _REGISTRY and type(_REGISTRY[name]) is not spec_cls:
        raise ValueError(f"モデル名 {name} は既に登録されています")
    _REGISTRY[name] = spec_cls()
    return spec_cls


def get_model_spec(name: str) -> ModelSpec:
    """モデル名から `ModelSpec` を取得する。

    Raises:
        KeyError: 未登録のモデル名の場合。
    """
    if name not in _REGISTRY:
        raise KeyError(f"未登録のモデルです: {name}（利用可能: {available_models()}）")
    return _REGISTRY[name]


def available_models() -> list[str]:
    """登録済みのモデル名の一覧を返す。"""
    return sorted(_REGISTRY)


def merge_params(defaults: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """既定パラメータにユーザー指定を上書きしたdictを返す。"""
    return {**defaults, **params}
