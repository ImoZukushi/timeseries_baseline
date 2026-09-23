"""日付・時刻の特徴量エンジニアリング（scikit-learn互換transformer）。

各クラスは `sklearn.base.BaseEstimator` / `TransformerMixin` を継承する。
本モジュールの変換は元の日時列は残したまま `{列名}_{接尾辞}` の形式で新しい
列を追加する（日時列そのものを上書きしない）。
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Sequence
from typing import Any, Literal, Self, cast

import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engineering._polars_sklearn import as_variable_list

_COMPONENT_EXPRS = {
    "year": lambda c: pl.col(c).dt.year(),
    "month": lambda c: pl.col(c).dt.month(),
    "day": lambda c: pl.col(c).dt.day(),
    "weekday": lambda c: pl.col(c).dt.weekday(),
    "week": lambda c: pl.col(c).dt.week(),
    "hour": lambda c: pl.col(c).dt.hour(),
    "minute": lambda c: pl.col(c).dt.minute(),
    "second": lambda c: pl.col(c).dt.second(),
}

_UNIT_SECONDS = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
}


class DatetimeFeaturesExtractor(BaseEstimator, TransformerMixin):
    """日時カラムから年・月・日・曜日・週・時・分・秒を抽出する（学習不要）。

    `weekday` は月曜=1〜日曜=7（ISO 8601）、`week` はISO週番号。
    生成される列名は `{列名}_{成分名}`（例: `ts_year`, `ts_month`）。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        components: Sequence[str] = (
            "year",
            "month",
            "day",
            "weekday",
            "week",
            "hour",
            "minute",
            "second",
        ),
    ) -> None:
        self.variables = variables
        self.components = components

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        unknown = set(self.components) - set(_COMPONENT_EXPRS)
        if unknown:
            raise ValueError(f"未知の成分が指定されました: {sorted(unknown)}")
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """日時成分を抽出して列を追加する。"""
        exprs = [
            _COMPONENT_EXPRS[component](col).alias(f"{col}_{component}")
            for col in self.variables_
            for component in self.components
        ]
        return X.with_columns(exprs)


class ElapsedTimeTransformer(BaseEstimator, TransformerMixin):
    """基準時刻からの経過時間を算出する（基準時刻は訓練データから学習）。

    `fit` 時に対象カラムの最小値を基準時刻として学習し、`transform` では
    `(対象時刻 - 基準時刻)` を指定単位の実数値に変換した列
    `{列名}_elapsed_{unit}` を追加する（テストデータが基準時刻より前でも
    負の値になるだけで、そのまま計算できる）。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        reference_: 各カラムの基準時刻（訓練データの最小値）。
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        unit: Literal["seconds", "minutes", "hours", "days"] = "days",
    ) -> None:
        self.variables = variables
        self.unit = unit

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """訓練データの各対象カラムの最小値を基準時刻として学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.reference_: dict[str, dt.datetime] = {
            col: cast(dt.datetime, X[col].min()) for col in self.variables_
        }
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """基準時刻からの経過時間（指定単位の実数）を列として追加する。"""
        divisor = _UNIT_SECONDS[self.unit]
        exprs = [
            (
                (pl.col(col) - self.reference_[col]).dt.total_microseconds() / 1_000_000 / divisor
            ).alias(f"{col}_elapsed_{self.unit}")
            for col in self.variables_
        ]
        return X.with_columns(exprs)


class CyclicalFeaturesEncoder(BaseEstimator, TransformerMixin):
    """周期特徴量（sin・cos）エンコーディング（学習不要）。

    周期性を持つ数値（月:12、時:24、曜日:7 等）を、境界の不連続性
    （12月の次が1月に戻る、23時の次が0時に戻る等）が生じないよう
    sin・cosの2列に変換する。周期 `period` は呼び出し側が指定する
    （データから自動推定はしない）。生成される列名は
    `{列名}_sin` / `{列名}_cos`。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
    """

    def __init__(self, variables: str | Sequence[str], period: float) -> None:
        self.variables = variables
        self.period = period

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """sin・cosの周期特徴量を列として追加する。"""
        exprs = []
        for col in self.variables_:
            angle = pl.col(col) * (2 * math.pi / self.period)
            exprs.append(angle.sin().alias(f"{col}_sin"))
            exprs.append(angle.cos().alias(f"{col}_cos"))
        return X.with_columns(exprs)
