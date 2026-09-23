"""時系列特徴量エンジニアリング（scikit-learn互換transformer）。

各クラスは `sklearn.base.BaseEstimator` / `TransformerMixin` を継承する。
本モジュールの変換はいずれも元の列は残したまま新しい列を追加する。
**リーク防止のため、未来方向の参照（中心化した移動平均や負のラグ等）は
一切サポートしない**（`LagFeatureGenerator` は正のラグのみを許可し、
`MovingAverageTransformer` は常に後方参照のみで中心化オプションを持たない）。

入力データは呼び出し側で対象の時系列順にソート済みであることを前提とする
（本モジュールはソートを行わない）。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engineering._polars_sklearn import as_variable_list


class LagFeatureGenerator(BaseEstimator, TransformerMixin):
    """ラグ特徴量（過去の値をずらして新しい列にする）。

    `lags` は全て正の整数（1以上）である必要がある。0以下を含む場合は
    未来のデータを参照することになるためfit時にValueErrorになる。
    生成される列名は `{列名}_lag_{n}`。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        lags_: fitで確定したラグ数のリスト。
    """

    def __init__(self, variables: str | Sequence[str], lags: Sequence[int]) -> None:
        self.variables = variables
        self.lags = lags

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """対象カラム・ラグ数を確定する（学習は不要だが妥当性を検証する）。"""
        self.variables_ = as_variable_list(self.variables)
        self.lags_ = list(self.lags)
        if not self.lags_:
            raise ValueError("lagsを1件以上指定してください。")
        if any(lag <= 0 for lag in self.lags_):
            raise ValueError(
                "lagsは全て正の整数（過去方向）である必要があります。"
                "未来のデータを参照するラグ（0以下）は許可していません。"
            )
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """各ラグの値を列として追加する（先頭 `lag` 件はnullになる）。"""
        exprs = [
            pl.col(col).shift(lag).alias(f"{col}_lag_{lag}")
            for col in self.variables_
            for lag in self.lags_
        ]
        return X.with_columns(exprs)


class MovingAverageTransformer(BaseEstimator, TransformerMixin):
    """移動平均（常に後方参照のみ。中心化オプションは提供しない）。

    生成される列名は `{列名}_ma_{window}`。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
    """

    def __init__(
        self, variables: str | Sequence[str], window: int, min_periods: int | None = None
    ) -> None:
        self.variables = variables
        self.window = window
        self.min_periods = min_periods

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """移動平均を列として追加する。"""
        exprs = [
            pl.col(col)
            .rolling_mean(self.window, min_samples=self.min_periods)
            .alias(f"{col}_ma_{self.window}")
            for col in self.variables_
        ]
        return X.with_columns(exprs)


class RateOfChangeTransformer(BaseEstimator, TransformerMixin):
    """変化率（前回比）。分母が0の場合はnullにする。

    生成される列名は `{列名}_roc_{periods}`。`inverse_transform` は
    `periods=1` の場合のみ対応する（累積積による絶対値の復元は、2点以上先の
    変化率では単一の `initial_value` から一意に復元できないため）。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
    """

    def __init__(self, variables: str | Sequence[str], periods: int = 1) -> None:
        self.variables = variables
        self.periods = periods

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """変化率を列として追加する。"""
        exprs = []
        for col in self.variables_:
            previous = pl.col(col).shift(self.periods)
            rate = pl.when(previous == 0).then(None).otherwise((pl.col(col) - previous) / previous)
            exprs.append(rate.alias(f"{col}_roc_{self.periods}"))
        return X.with_columns(exprs)

    def inverse_transform(self, X: pl.DataFrame, initial_value: float) -> pl.DataFrame:
        """変化率列（`periods=1`）から絶対値を復元し、元のカラム名で列を追加/置換する。

        先頭行以降のnull（0除算により変化率が定義できなかった行）は
        「変化なし（変化率0）」とみなして復元する。これは、値が一度0になった
        直後の変化率は数学的に定義できない（0からの相対変化を有限の比では
        表現できない）ためで、このケースでは往復変換をしても元の値には
        戻らない点に注意（これはtransform/inverse_transformの実装上の不備ではなく、
        変化率という表現方法そのものの原理的な限界）。0を経由しないデータでは
        往復変換で厳密に元の値へ戻る。

        Args:
            X: `{列名}_roc_1` 列を含むDataFrame。
            initial_value: 復元の起点となる最初の絶対値（先頭行の変化率はnullのため
                この値がそのまま先頭行の復元値になる）。

        Returns:
            復元した絶対値の列（元のカラム名）を追加/置換したDataFrame。

        Raises:
            NotImplementedError: `periods != 1` の場合。
        """
        if self.periods != 1:
            raise NotImplementedError("inverse_transformはperiods=1の変化率にのみ対応しています。")
        out = X
        for col in self.variables_:
            rate_col = f"{col}_roc_{self.periods}"
            growth_factor = 1.0 + X[rate_col].fill_null(0.0)
            reconstructed = growth_factor.cum_prod() * initial_value
            out = out.with_columns(reconstructed.alias(col))
        return out
