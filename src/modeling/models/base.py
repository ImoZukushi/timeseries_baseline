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

from evaluation.curves import TrainingHistory
from modeling.tasks import Task

ExplainerKind = Literal["tree", "linear", "permutation"]
CategoricalEncoding = Literal["ordinal", "onehot"]

# 回帰の目的関数（損失）の共通の名前。各モデルの指定への変換は `ModelSpec.objective_params` で行う
#   l2: 二乗誤差 / l1: 絶対誤差 / huber: Huber損失 / quantile: 分位点損失 / poisson: ポアソン損失
OBJECTIVES: tuple[str, ...] = ("l2", "l1", "huber", "quantile", "poisson")

# `alpha` を省略したときの既定値。モデルごとの既定（LightGBM は huber・quantile とも 0.9 など）が
# ばらばらなので、モデルを替えても同じ損失になるよう共通の値にそろえる
DEFAULT_OBJECTIVE_ALPHA: dict[str, float] = {"huber": 1.0, "quantile": 0.5}


class ModelSpec(ABC):
    """モデル1種類分の定義。

    Attributes:
        name: YAMLの `model.name` で指定する名前。
        supported_tasks: 対応するタスク。
        explainer_kind: SHAPで使うexplainerの種類。
        requires_float32: 入力特徴量をfloat32に変換する必要があるか（PyTorch系）。
        shap_max_samples: SHAPを計算する行数の上限（Noneなら設定の `explain.max_samples` のまま）。
            1行あたりの計算が重いモデル（深い木を多数持つランダムフォレスト系）で使う。
        categorical_encoding: カテゴリ列の推奨のエンコーディング。`"ordinal"`（整数化。木のモデル
            向き）か `"onehot"`（ワンホット。整数の大小に意味を持たせない線形モデル向き）。
            設定ファイル・ノートブックが特徴量の前処理を選ぶときの目安で、モデル自体は使わない。
    """

    name: ClassVar[str]
    supported_tasks: ClassVar[frozenset[Task]]
    explainer_kind: ClassVar[ExplainerKind] = "permutation"
    requires_float32: ClassVar[bool] = False
    shap_max_samples: ClassVar[int | None] = None
    categorical_encoding: ClassVar[CategoricalEncoding] = "ordinal"

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

    @property
    def has_search_space(self) -> bool:
        """既定の探索空間を持つか（`search_space` を実装しているか）。

        ハイパーパラメータの無いモデル（OLS・定数予測）は False。Optuna で探す前の確認に使う。
        """
        return type(self).search_space is not ModelSpec.search_space

    def fit_kwargs(
        self,
        X_valid: Any,
        y_valid: Any,
        early_stopping_rounds: int | None,
        *,
        X_train: Any = None,
        y_train: Any = None,
        record_history: bool = False,
    ) -> dict[str, Any]:
        """`estimator.fit` に渡す追加引数（early stopping用の検証データ等）を返す。

        Args:
            X_valid: 検証データ（前処理後。無ければNone）。
            y_valid: 検証データの目的変数。
            early_stopping_rounds: early stoppingのラウンド数（Noneなら無効）。
            X_train: 学習データ（前処理後）。学習の推移を記録する場合に評価セットとして使う。
            y_train: 学習データの目的変数。
            record_history: Trueなら学習・検証の損失を反復ごとに記録する
                （`training_history` で取り出す）。

        既定では追加引数なし。
        """
        return {}

    def training_history(self, estimator: Any) -> TrainingHistory | None:
        """学習済みestimatorから、反復ごとの学習・検証の損失を取り出す（記録が無ければNone）。"""
        return None

    def best_iteration(self, estimator: Any) -> int | None:
        """early stoppingで決まった最良イテレーション数を返す（該当しなければNone）。"""
        return None

    def with_n_iterations(self, params: dict[str, Any], n_iterations: int) -> dict[str, Any]:
        """全データ再学習時に、イテレーション数を固定したパラメータを返す。

        既定ではパラメータをそのまま返す（イテレーションの概念がないモデル）。
        """
        return params

    def objective_params(self, objective: str, alpha: float | None = None) -> dict[str, Any]:
        """共通の名前（`OBJECTIVES`）の目的関数を、このモデルのパラメータに変換する（回帰用）。

        Args:
            objective: 目的関数の共通の名前（`l2`・`l1`・`huber`・`quantile`・`poisson`）。
            alpha: huber の閾値（誤差がこれを超えると絶対誤差になる）・quantile の分位
                （Noneなら `DEFAULT_OBJECTIVE_ALPHA`）。l2・l1・poisson では使わない。

        Returns:
            モデルのパラメータに加える dict。

        Raises:
            ValueError: 未知の名前、またはこのモデルが対応しない目的関数の場合。
        """
        check_objective(objective)
        raise unsupported_objective(self, objective)

    def supports(self, task: Task) -> bool:
        """タスクに対応しているかを返す。"""
        return task in self.supported_tasks


def check_objective(objective: str) -> None:
    """目的関数の共通の名前が `OBJECTIVES` にあるかを確かめる。

    Raises:
        ValueError: 未知の名前の場合。
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"未知の目的関数です: {objective}（利用可能: {list(OBJECTIVES)}）")


def objective_alpha(objective: str, alpha: float | None) -> float:
    """huber・quantile の `alpha`（省略時は `DEFAULT_OBJECTIVE_ALPHA`）を返す。

    Raises:
        ValueError: quantile の分位が (0, 1) の外、または huber の閾値が正でない場合。
    """
    value = DEFAULT_OBJECTIVE_ALPHA[objective] if alpha is None else float(alpha)
    if objective == "quantile" and not 0 < value < 1:
        raise ValueError(f"quantile の分位は 0 < alpha < 1 で指定してください: {value}")
    if objective == "huber" and value <= 0:
        raise ValueError(f"huber の閾値は正の値で指定してください: {value}")
    return value


def unsupported_objective(spec: ModelSpec, objective: str) -> ValueError:
    """このモデルが対応しない目的関数を指定されたときの例外を作る。"""
    return ValueError(f"{spec.name} は目的関数 {objective} に対応していません")


_REGISTRY: dict[str, ModelSpec] = {}
# 任意依存が無いなどの理由で登録できなかったモデル（モデル名 → 理由）
_UNAVAILABLE: dict[str, str] = {}


def register_unavailable(names: tuple[str, ...] | list[str], reason: str) -> None:
    """登録できなかったモデルと、その理由を記録する（`get_model_spec` のエラーに出す）。

    任意依存（torch など）を import できない環境で、モデル名の書き間違いと区別できるようにする。

    Args:
        names: 登録できなかったモデル名。
        reason: 理由（インストール方法など）。
    """
    for name in names:
        _UNAVAILABLE[name] = reason


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
        KeyError: 未登録のモデル名の場合（任意依存が無くて登録できなかったモデルなら、
            その理由も出す）。
    """
    if name not in _REGISTRY:
        if name in _UNAVAILABLE:
            raise KeyError(f"モデル {name} はこの環境では使えません: {_UNAVAILABLE[name]}")
        raise KeyError(f"未登録のモデルです: {name}（利用可能: {available_models()}）")
    return _REGISTRY[name]


def available_models() -> list[str]:
    """登録済みのモデル名の一覧を返す。"""
    return sorted(_REGISTRY)


def merge_params(defaults: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """既定パラメータにユーザー指定を上書きしたdictを返す。"""
    return {**defaults, **params}
