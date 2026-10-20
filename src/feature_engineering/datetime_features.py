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

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from feature_engineering._pandas_sklearn import as_variable_list, replace_columns

_COMPONENT_EXPRS = {
    "year": lambda s: s.dt.year,
    "month": lambda s: s.dt.month,
    "day": lambda s: s.dt.day,
    # 月曜=1〜日曜=7（ISO 8601。pandas の dayofweek は月曜=0）
    "weekday": lambda s: s.dt.dayofweek + 1,
    "week": lambda s: s.dt.isocalendar().week,
    "hour": lambda s: s.dt.hour,
    "minute": lambda s: s.dt.minute,
    "second": lambda s: s.dt.second,
}


def _as_number(values: pd.Series) -> pd.Series:
    """日時の成分を数値の列にする（欠損があれば float、無ければ int64）。"""
    if values.isna().any():
        return values.astype("float64")
    return values.astype("int64")


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

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        unknown = set(self.components) - set(_COMPONENT_EXPRS)
        if unknown:
            raise ValueError(f"未知の成分が指定されました: {sorted(unknown)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """日時成分を抽出して列を追加する。"""
        values = {}
        for col in self.variables_:
            ts = pd.to_datetime(X[col])
            for component in self.components:
                values[f"{col}_{component}"] = _as_number(_COMPONENT_EXPRS[component](ts))
        return replace_columns(X, values)


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

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """訓練データの各対象カラムの最小値を基準時刻として学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.reference_: dict[str, dt.datetime] = {
            col: cast(dt.datetime, pd.Timestamp(X[col].min()).to_pydatetime())
            for col in self.variables_
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """基準時刻からの経過時間（指定単位の実数）を列として追加する。"""
        divisor = _UNIT_SECONDS[self.unit]
        values = {
            f"{col}_elapsed_{self.unit}": (
                (pd.to_datetime(X[col]) - pd.Timestamp(self.reference_[col])).dt.total_seconds()
                / divisor
            )
            for col in self.variables_
        }
        return replace_columns(X, values)


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

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """対象カラムを確定する（学習は不要）。"""
        self.variables_ = as_variable_list(self.variables)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """sin・cosの周期特徴量を列として追加する。"""
        values = {}
        for col in self.variables_:
            angle = X[col].astype(float) * (2 * math.pi / self.period)
            values[f"{col}_sin"] = np.sin(angle)
            values[f"{col}_cos"] = np.cos(angle)
        return replace_columns(X, values)
