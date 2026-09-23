"""数値変数の特徴量エンジニアリング（scikit-learn互換transformer）。

各クラスは `sklearn.base.BaseEstimator` / `TransformerMixin` を継承する。
本モジュールの変換は対象列を同名のまま置き換える（列を追加はしない）。
`LogTransformer` 等の学習不要な変換も、`sklearn.pipeline.Pipeline` に
組み込めるよう `fit` を実装しているが、中身は対象カラムの記録のみで
実質的にno-op（訓練データの値そのものは記憶しない）。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Self

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

from feature_engineering._polars_sklearn import as_variable_list, to_numpy_2d


class LogTransformer(BaseEstimator, TransformerMixin):
    """対数変換。0以下の値はnullにする。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
    """

    def __init__(self, variables: str | Sequence[str], base: float = math.e) -> None:
        self.variables = variables
        self.base = base

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """対数変換を適用する。"""
        exprs = [
            pl.when(pl.col(col) > 0).then(pl.col(col).log(self.base)).otherwise(None).alias(col)
            for col in self.variables_
        ]
        return X.with_columns(exprs)

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """対数変換の逆変換（累乗）を適用する。"""
        exprs = [(self.base ** pl.col(col)).alias(col) for col in self.variables_]
        return X.with_columns(exprs)


class ReciprocalTransformer(BaseEstimator, TransformerMixin):
    """逆数変換（1/x）。x=0はnullにする。逆変換も同じ式（1/(1/x)=x）になる。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
    """

    def __init__(self, variables: str | Sequence[str]) -> None:
        self.variables = variables

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """逆数変換を適用する。"""
        exprs = [
            pl.when(pl.col(col) != 0).then(1.0 / pl.col(col)).otherwise(None).alias(col)
            for col in self.variables_
        ]
        return X.with_columns(exprs)

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """逆数変換の逆変換（再度逆数を取る）を適用する。"""
        return self.transform(X)


class SqrtTransformer(BaseEstimator, TransformerMixin):
    """平方根変換。負値はnullにする。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
    """

    def __init__(self, variables: str | Sequence[str]) -> None:
        self.variables = variables

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """平方根変換を適用する。"""
        exprs = [
            pl.when(pl.col(col) >= 0).then(pl.col(col).sqrt()).otherwise(None).alias(col)
            for col in self.variables_
        ]
        return X.with_columns(exprs)

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """平方根変換の逆変換（2乗）を適用する。"""
        exprs = [(pl.col(col) ** 2).alias(col) for col in self.variables_]
        return X.with_columns(exprs)


class FixedPowerTransformer(BaseEstimator, TransformerMixin):
    """固定指数によるべき乗変換（学習不要な変換であり、Box-Cox/Yeo-Johnsonとは異なる）。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
    """

    def __init__(self, variables: str | Sequence[str], power: float) -> None:
        self.variables = variables
        self.power = power

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """べき乗変換を適用する。"""
        exprs = [(pl.col(col) ** self.power).alias(col) for col in self.variables_]
        return X.with_columns(exprs)

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """べき乗変換の逆変換（1/power乗）を適用する。"""
        exprs = [(pl.col(col) ** (1.0 / self.power)).alias(col) for col in self.variables_]
        return X.with_columns(exprs)


class PolarsPowerTransformer(BaseEstimator, TransformerMixin):
    """Box-Cox変換・Yeo-Johnson変換（`sklearn.preprocessing.PowerTransformer` をラップ）。

    λ（分布を最も正規分布に近づけるパラメータ）を訓練データから学習する。
    Box-Coxは正の値のみに適用可能で、0以下を含む訓練データでfitすると
    scikit-learn側でValueErrorになる（その場合はYeo-Johnsonを使う）。
    スケーリング（標準化）は別カテゴリのため行わない（`standardize=False`）。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        transformer_: fit済みの内部 `PowerTransformer`。
    """

    def __init__(self, variables: str | Sequence[str], method: str = "yeo-johnson") -> None:
        self.variables = variables
        self.method = method

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """訓練データからλを学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.transformer_ = PowerTransformer(method=self.method, standardize=False)
        self.transformer_.fit(to_numpy_2d(X, self.variables_))
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """学習済みのλで変換する。"""
        transformed = self.transformer_.transform(to_numpy_2d(X, self.variables_))
        replaced = [pl.Series(col, transformed[:, i]) for i, col in enumerate(self.variables_)]
        return X.with_columns(replaced)

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """学習済みのλで逆変換する。"""
        original = self.transformer_.inverse_transform(to_numpy_2d(X, self.variables_))
        replaced = [pl.Series(col, original[:, i]) for i, col in enumerate(self.variables_)]
        return X.with_columns(replaced)
