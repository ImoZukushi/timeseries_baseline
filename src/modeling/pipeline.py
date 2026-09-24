"""設定から sklearn Pipeline（特徴量エンジニアリング → モデル入力変換 → モデル）を組み立てる。

前処理はpolars DataFrameのまま行い、モデル直前の `ToModelInput` でpandas DataFrameに
変換する。pandasを使うのは、LightGBM・XGBoost・SHAPが特徴量名をpandasの列名から
取得するため（モデル境界でのみ使用する）。
"""

from __future__ import annotations

import importlib
from typing import Any, Self

import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from modeling.config import ExperimentConfig, FeatureStepConfig
from modeling.models import get_model_spec

MODEL_STEP = "model"


class ToModelInput(BaseEstimator, TransformerMixin):
    """polars DataFrameをモデル入力用のpandas DataFrameに変換する。

    数値型（整数・浮動小数・真偽値）以外の列が残っている場合は、どの列を
    エンコードすべきかが分かるよう列名を列挙してエラーにする
    （文字列や日時をモデルに暗黙に渡して意図しない扱いになるのを防ぐ）。
    `fit` 時の列順を記録し、`transform` では同じ列順に揃える。

    Args:
        dtype: 出力のdtype（PyTorch系モデルでは `"float32"`）。Noneなら元の数値型のまま。

    Attributes:
        feature_names_in_: fit時の列名。
    """

    def __init__(self, dtype: str | None = None) -> None:
        self.dtype = dtype

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """列名を記録し、数値以外の列が無いことを確認する。

        Raises:
            TypeError: 数値型以外の列が含まれる場合。
        """
        non_numeric = [
            name
            for name, dtype in X.schema.items()
            if not (dtype.is_numeric() or dtype == pl.Boolean)
        ]
        if non_numeric:
            raise TypeError(
                "モデルに渡す特徴量に数値型以外の列があります。features でエンコードするか "
                f"data.drop_cols で除外してください: {non_numeric}"
            )
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pl.DataFrame) -> pd.DataFrame:
        """fit時の列順でpandas DataFrameに変換する。"""
        # polarsのnullはpandas側でNaNとして扱われるようfloat変換可能な形で出力する
        frame = X.select(list(self.feature_names_in_)).to_pandas()
        if self.dtype is not None:
            frame = frame.astype(self.dtype)  # type: ignore[arg-type]
        return frame

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """出力列名（fit時の列名）を返す。"""
        return self.feature_names_in_


class DropColumns(BaseEstimator, TransformerMixin):
    """指定した列を削除する（学習不要）。

    ワンホットエンコーディングや日時成分抽出のように「元の列を残して新しい列を追加する」
    transformerの後段で、モデルに渡さない元の列を取り除くために使う。

    Args:
        columns: 削除する列名のリスト。

    Attributes:
        columns_: fitで確定した削除対象の列名。
    """

    def __init__(self, columns: str | list[str]) -> None:
        self.columns = columns

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """削除対象の列が存在することを確認する。

        Raises:
            KeyError: 存在しない列が指定された場合。
        """
        self.columns_ = [self.columns] if isinstance(self.columns, str) else list(self.columns)
        missing = [c for c in self.columns_ if c not in X.columns]
        if missing:
            raise KeyError(f"削除対象の列がありません: {missing}")
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """指定列を削除したDataFrameを返す。"""
        return X.drop(self.columns_)


def import_class(class_path: str) -> type:
    """dotted path（例: `feature_engineering.numeric.LogTransformer`）からクラスを取得する。

    Raises:
        ImportError: モジュールまたはクラスが見つからない場合。
    """
    module_name, _, class_name = class_path.rpartition(".")
    if not module_name:
        raise ImportError(f"クラスはモジュール名付きで指定してください: {class_path}")
    module = importlib.import_module(module_name)
    try:
        cls: type = getattr(module, class_name)
    except AttributeError as e:
        raise ImportError(f"{module_name} に {class_name} がありません") from e
    return cls


def build_feature_steps(steps: list[FeatureStepConfig]) -> list[tuple[str, Any]]:
    """特徴量エンジニアリングのステップをPipeline用の `(名前, transformer)` に変換する。"""
    built = []
    for i, step in enumerate(steps):
        cls = import_class(step.class_path)
        built.append((f"{i:02d}_{cls.__name__}", cls(**step.params)))
    return built


def build_pipeline(config: ExperimentConfig, params: dict[str, Any] | None = None) -> Pipeline:
    """実験設定から未学習のPipelineを作る。

    Args:
        config: 実験設定。
        params: モデルパラメータ（チューニング結果等）。Noneなら `config.model.params`。

    Returns:
        `[特徴量ステップ..., ("to_model_input", ToModelInput), ("model", estimator)]` のPipeline。
    """
    spec = get_model_spec(config.model.name)
    if not spec.supports(config.task):
        raise ValueError(f"モデル {spec.name} はタスク {config.task} に対応していません")
    model = spec.build(
        config.task,
        config.model.params if params is None else params,
        seed=config.seed,
        early_stopping_rounds=config.model.early_stopping_rounds,
    )
    to_input = ToModelInput(dtype="float32" if spec.requires_float32 else None)
    return Pipeline(
        [*build_feature_steps(config.features), ("to_model_input", to_input), (MODEL_STEP, model)]
    )
