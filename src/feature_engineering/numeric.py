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

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer

from feature_engineering._pandas_sklearn import as_variable_list, replace_columns, to_numpy_2d


class LogTransformer(BaseEstimator, TransformerMixin):
    """対数変換 `log(x + offset)`。`x + offset` が0以下の値は欠損（NaN）にする。

    Args:
        variables: 対象カラム名。
        base: 対数の底（既定は自然対数）。
        offset: 対数を取る前に足す値。0を含む系列（件数など）では `offset=1`
            （`log(1 + x)`）のように指定する。既定0で通常の対数変換。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
    """

    def __init__(
        self, variables: str | Sequence[str], base: float = math.e, offset: float = 0.0
    ) -> None:
        self.variables = variables
        self.base = base
        self.offset = offset

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """対数変換を適用する。"""
        values = {}
        for col in self.variables_:
            shifted = X[col].astype(float) + self.offset
            # 0以下は対数が定義されないため欠損にする
            values[col] = (np.log(shifted.where(shifted > 0)) / np.log(self.base)).astype(float)
        return replace_columns(X, values)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """対数変換の逆変換（`base ** y - offset`）を適用する。"""
        return replace_columns(
            X, {col: self.base ** X[col].astype(float) - self.offset for col in self.variables_}
        )


class ReciprocalTransformer(BaseEstimator, TransformerMixin):
    """逆数変換（1/x）。x=0は欠損（NaN）にする。逆変換も同じ式（1/(1/x)=x）になる。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
    """

    def __init__(self, variables: str | Sequence[str]) -> None:
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """逆数変換を適用する。"""
        values = {}
        for col in self.variables_:
            x = X[col].astype(float)
            values[col] = 1.0 / x.where(x != 0)  # 0は欠損にする
        return replace_columns(X, values)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """逆数変換の逆変換（再度逆数を取る）を適用する。"""
        return self.transform(X)


class SqrtTransformer(BaseEstimator, TransformerMixin):
    """平方根変換。負値は欠損（NaN）にする。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
    """

    def __init__(self, variables: str | Sequence[str]) -> None:
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """平方根変換を適用する。"""
        values = {}
        for col in self.variables_:
            x = X[col].astype(float)
            values[col] = np.sqrt(x.where(x >= 0))  # 負値は欠損にする
        return replace_columns(X, values)

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """平方根変換の逆変換（2乗）を適用する。"""
        return replace_columns(X, {col: X[col].astype(float) ** 2 for col in self.variables_})


class FixedPowerTransformer(BaseEstimator, TransformerMixin):
    """固定指数によるべき乗変換（学習不要な変換であり、Box-Cox/Yeo-Johnsonとは異なる）。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
    """

    def __init__(self, variables: str | Sequence[str], power: float) -> None:
        self.variables = variables
        self.power = power

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """べき乗変換を適用する。"""
        return replace_columns(
            X, {col: X[col].astype(float) ** self.power for col in self.variables_}
        )

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """べき乗変換の逆変換（1/power乗）を適用する。"""
        return replace_columns(
            X, {col: X[col].astype(float) ** (1.0 / self.power) for col in self.variables_}
        )


class PandasPowerTransformer(BaseEstimator, TransformerMixin):
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

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """訓練データからλを学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.transformer_ = PowerTransformer(method=self.method, standardize=False)
        self.transformer_.fit(to_numpy_2d(X, self.variables_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """学習済みのλで変換する。"""
        transformed = self.transformer_.transform(to_numpy_2d(X, self.variables_))
        return replace_columns(X, {col: transformed[:, i] for i, col in enumerate(self.variables_)})

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """学習済みのλで逆変換する。"""
        original = self.transformer_.inverse_transform(to_numpy_2d(X, self.variables_))
        return replace_columns(X, {col: original[:, i] for i, col in enumerate(self.variables_)})
