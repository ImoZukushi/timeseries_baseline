"""スケーリング（正規化）の特徴量エンジニアリング（scikit-learn互換transformer）。

各クラスは `sklearn.base.BaseEstimator` / `TransformerMixin` を継承する。
本モジュールの変換はいずれも対象列を同名のまま置き換え、全て `inverse_transform`
（予測値等を元のスケールに戻す用途）を実装する。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler

from feature_engineering._polars_sklearn import as_variable_list, to_numpy_2d


class PolarsMinMaxScaler(BaseEstimator, TransformerMixin):
    """最大値・最小値によるスケーリング（`sklearn.preprocessing.MinMaxScaler` をラップ）。

    `(x - min) / (max - min)` で[0, 1]区間に変換する。訓練データが定数列
    （max == min）の場合はscikit-learn側の仕様により0を返す（0除算にはならない）。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        scaler_: fit済みの内部 `MinMaxScaler`。
    """

    def __init__(self, variables: str | Sequence[str]) -> None:
        self.variables = variables

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """訓練データから最大値・最小値を学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(to_numpy_2d(X, self.variables_))
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """学習済みの最大値・最小値でスケーリングする。"""
        scaled = self.scaler_.transform(to_numpy_2d(X, self.variables_))
        return X.with_columns(
            [pl.Series(col, scaled[:, i]) for i, col in enumerate(self.variables_)]
        )

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """元のスケールに戻す。"""
        original = self.scaler_.inverse_transform(to_numpy_2d(X, self.variables_))
        return X.with_columns(
            [pl.Series(col, original[:, i]) for i, col in enumerate(self.variables_)]
        )


class PolarsRobustScaler(BaseEstimator, TransformerMixin):
    """中央値・分位数によるロバストスケーリング（`sklearn.preprocessing.RobustScaler` をラップ）。

    `(x - median) / (Q_high - Q_low)` で変換する。外れ値の影響を受けにくい。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        scaler_: fit済みの内部 `RobustScaler`。
    """

    def __init__(
        self, variables: str | Sequence[str], quantile_range: tuple[float, float] = (25.0, 75.0)
    ) -> None:
        self.variables = variables
        self.quantile_range = quantile_range

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """訓練データから中央値・分位数を学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.scaler_ = RobustScaler(quantile_range=self.quantile_range)
        self.scaler_.fit(to_numpy_2d(X, self.variables_))
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """学習済みの中央値・分位数でスケーリングする。"""
        scaled = self.scaler_.transform(to_numpy_2d(X, self.variables_))
        return X.with_columns(
            [pl.Series(col, scaled[:, i]) for i, col in enumerate(self.variables_)]
        )

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """元のスケールに戻す。"""
        original = self.scaler_.inverse_transform(to_numpy_2d(X, self.variables_))
        return X.with_columns(
            [pl.Series(col, original[:, i]) for i, col in enumerate(self.variables_)]
        )


class MeanNormalizationScaler(BaseEstimator, TransformerMixin):
    """平均正規化（`(x - mean) / (max - min)`、scikit-learnに相当品なし）。

    標準化（z-score, `(x-mean)/std`）とは異なり、分母に標準偏差ではなく
    値域（max-min）を使う。訓練データが定数列（max == min）の場合は
    0除算を避けるため0を返す。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        mean_: 各カラムの平均値（訓練データ）。
        min_: 各カラムの最小値（訓練データ）。
        max_: 各カラムの最大値（訓練データ）。
    """

    def __init__(self, variables: str | Sequence[str]) -> None:
        self.variables = variables

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """訓練データから平均値・最小値・最大値を学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.mean_: dict[str, float] = {col: float(X[col].mean()) for col in self.variables_}  # type: ignore[arg-type]
        self.min_: dict[str, float] = {col: float(X[col].min()) for col in self.variables_}  # type: ignore[arg-type]
        self.max_: dict[str, float] = {col: float(X[col].max()) for col in self.variables_}  # type: ignore[arg-type]
        return self

    def _scale(self, col: str) -> float:
        scale = self.max_[col] - self.min_[col]
        return scale if scale != 0 else 1.0

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """学習済みの平均値・値域で平均正規化する。"""
        exprs = [
            ((pl.col(col) - self.mean_[col]) / self._scale(col)).alias(col)
            for col in self.variables_
        ]
        return X.with_columns(exprs)

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """元のスケールに戻す。"""
        exprs = [
            (pl.col(col) * self._scale(col) + self.mean_[col]).alias(col) for col in self.variables_
        ]
        return X.with_columns(exprs)


class PolarsMaxAbsScaler(BaseEstimator, TransformerMixin):
    """最大絶対値スケーリング（`sklearn.preprocessing.MaxAbsScaler` をラップ）。

    `x / max(|x|)` で[-1, 1]区間に変換する。訓練データが全て0の場合は
    scikit-learn側の仕様によりそのまま0を返す（0除算にはならない）。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        scaler_: fit済みの内部 `MaxAbsScaler`。
    """

    def __init__(self, variables: str | Sequence[str]) -> None:
        self.variables = variables

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """訓練データから最大絶対値を学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.scaler_ = MaxAbsScaler()
        self.scaler_.fit(to_numpy_2d(X, self.variables_))
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """学習済みの最大絶対値でスケーリングする。"""
        scaled = self.scaler_.transform(to_numpy_2d(X, self.variables_))
        return X.with_columns(
            [pl.Series(col, scaled[:, i]) for i, col in enumerate(self.variables_)]
        )

    def inverse_transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """元のスケールに戻す。"""
        original = self.scaler_.inverse_transform(to_numpy_2d(X, self.variables_))
        return X.with_columns(
            [pl.Series(col, original[:, i]) for i, col in enumerate(self.variables_)]
        )
