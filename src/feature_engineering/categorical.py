"""カテゴリ変数の特徴量エンジニアリング（scikit-learn互換transformer）。

各クラスは `sklearn.base.BaseEstimator` / `TransformerMixin` を継承しており、
`fit(X, y=None) -> self` → `transform(X) -> X`（`fit_transform`は`TransformerMixin`が提供）
という標準的なscikit-learnの流儀に従う。`sklearn.pipeline.Pipeline` /
`sklearn.compose.ColumnTransformer` にそのまま組み込める。

**データリーク防止の指針**: `fit` は必ず訓練データのみに対して呼び、検証・テスト
データには `transform`（`fit_transform`ではない）のみを呼ぶこと。`fit`で学習した
パラメータは `xxx_` という名前の属性に保存され、`transform` はそれらのみを参照して
検証・テストデータそのものの分布を一切参照しない。

**列の扱い方針**: 各カテゴリをそのままエンコードする技術（序数・ターゲット・
低頻度グループ化）は元の列を同名のまま置き換える。ワンホットエンコーディングのように
1列から複数列を生成する技術は、元の列は残したまま `{列名}_{カテゴリ名}` の形式で
新しい列を追加する。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, TargetEncoder

from feature_engineering._polars_sklearn import as_variable_list, to_numpy_2d


class PolarsOneHotEncoder(BaseEstimator, TransformerMixin):
    """ワンホットエンコーディング（`sklearn.preprocessing.OneHotEncoder` をラップ）。

    `min_frequency` / `max_categories` を指定しない場合は全カテゴリを対象とする。
    指定した場合は頻出カテゴリのみを個別の列にし、それ以外は
    `{列名}_infrequent_sklearn` としてまとめる（scikit-learnの標準挙動）。
    学習時に存在しなかったカテゴリは全列0になる。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        encoder_: fit済みの内部 `OneHotEncoder`。
        feature_names_out_: transformで生成される列名。
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        min_frequency: int | float | None = None,
        max_categories: int | None = None,
    ) -> None:
        self.variables = variables
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """訓練データからワンホットエンコーディングの対象カテゴリを学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.encoder_ = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            min_frequency=self.min_frequency,
            max_categories=self.max_categories,
        )
        self.encoder_.fit(to_numpy_2d(X, self.variables_))
        self.feature_names_out_ = list(self.encoder_.get_feature_names_out(self.variables_))
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """学習済みのカテゴリに基づきワンホット列を追加する。"""
        encoded = self.encoder_.transform(to_numpy_2d(X, self.variables_))
        new_columns = [
            pl.Series(name, encoded[:, i]) for i, name in enumerate(self.feature_names_out_)
        ]
        return X.with_columns(new_columns)


class PolarsOrdinalEncoder(BaseEstimator, TransformerMixin):
    """序数エンコーディング（整数を割当、`sklearn.preprocessing.OrdinalEncoder` をラップ）。

    カテゴリをソートした順に0,1,2...を割り当てる。学習時に存在しなかったカテゴリは
    nullになる。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        encoder_: fit済みの内部 `OrdinalEncoder`。
    """

    def __init__(self, variables: str | Sequence[str]) -> None:
        self.variables = variables

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """訓練データからカテゴリ→整数の対応を学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.encoder_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        self.encoder_.fit(to_numpy_2d(X, self.variables_))
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """学習済みの対応表でカテゴリを整数に置き換える。"""
        encoded = self.encoder_.transform(to_numpy_2d(X, self.variables_))
        replaced = [
            pl.Series(col, encoded[:, i]).fill_nan(None) for i, col in enumerate(self.variables_)
        ]
        return X.with_columns(replaced)


class TargetGuidedOrdinalEncoder(BaseEstimator, TransformerMixin):
    """目的変数平均でランク付けする序数エンコーディング（scikit-learnに相当品なし）。

    訓練データにおけるカテゴリ別の目的変数平均を計算し、その値が小さい順に
    0,1,2...の順位を割り当てる（値そのものではなく順位を使う点で、後述の
    `PolarsTargetEncoder` とは異なる）。学習時に存在しなかったカテゴリはnullになる。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        mappings_: 各カラムの「カテゴリ→順位」の対応表。
    """

    def __init__(self, variables: str | Sequence[str]) -> None:
        self.variables = variables

    def fit(self, X: pl.DataFrame, y: pl.Series | np.ndarray | Sequence[float]) -> Self:
        """訓練データとその目的変数から、カテゴリ別の目的変数平均順位を学習する。

        Args:
            X: 訓練データ。
            y: 目的変数（`X` と同じ行数）。

        Returns:
            self。
        """
        self.variables_ = as_variable_list(self.variables)
        target = pl.Series("__target__", y)
        self.mappings_: dict[str, dict[Any, int]] = {}
        for col in self.variables_:
            ranked = (
                pl.DataFrame({col: X[col], "__target__": target})
                .group_by(col)
                .agg(pl.col("__target__").mean().alias("__mean__"))
                .sort("__mean__")
            )
            self.mappings_[col] = {
                category: rank for rank, category in enumerate(ranked[col].to_list())
            }
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """学習済みの順位でカテゴリを置き換える。"""
        out = X
        for col in self.variables_:
            out = out.with_columns(
                pl.col(col)
                .replace_strict(self.mappings_[col], default=None, return_dtype=pl.Int64)
                .alias(col)
            )
        return out


class PolarsTargetEncoder(BaseEstimator, TransformerMixin):
    """ターゲットエンコーディング（`sklearn.preprocessing.TargetEncoder` をラップ）。

    カテゴリを目的変数平均の値そのもので置き換える。訓練データ自身に対して
    このエンコーディングを適用すると、自分の目的変数由来の統計量を自分の特徴量に
    使うことになり別種のデータリークが起きるため、`fit_transform` は
    scikit-learnの内部cross-fitting機構（`cv`分割し、各行を他foldの統計量で
    エンコードする）を使う。検証・テストデータには通常どおり
    `fit`（訓練データで）→`transform`（検証・テストデータに）を使う。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        encoder_: fit済みの内部 `TargetEncoder`。
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        smoothing: float = 0.0,
        cv: int = 5,
        target_type: str = "continuous",
        random_state: int = 0,
    ) -> None:
        self.variables = variables
        self.smoothing = smoothing
        self.cv = cv
        self.target_type = target_type
        self.random_state = random_state

    def _build_encoder(self) -> TargetEncoder:
        # cvに単なる整数を渡すと、scikit-learn側が内部で生成するfold分割が
        # シードされておらず実行のたびに結果が変わってしまう（再現性が無い）ため、
        # 明示的にシード付きのKFoldを渡して常に同じfold分割になるようにする。
        splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        return TargetEncoder(smooth=self.smoothing, cv=splitter, target_type=self.target_type)

    def fit(self, X: pl.DataFrame, y: pl.Series | np.ndarray | Sequence[float]) -> Self:
        """訓練データ全体から、カテゴリ別の目的変数平均を学習する（検証・テスト用）。"""
        self.variables_ = as_variable_list(self.variables)
        self.encoder_ = self._build_encoder()
        self.encoder_.fit(to_numpy_2d(X, self.variables_), np.asarray(y))
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """学習済みの目的変数平均でカテゴリを置き換える（未知カテゴリは学習時の全体平均）。"""
        encoded = self.encoder_.transform(to_numpy_2d(X, self.variables_))
        replaced = [pl.Series(col, encoded[:, i]) for i, col in enumerate(self.variables_)]
        return X.with_columns(replaced)

    def fit_transform(
        self, X: pl.DataFrame, y: pl.Series | np.ndarray | Sequence[float] | None = None, **_: Any
    ) -> pl.DataFrame:
        """訓練データ自身に安全に使えるcross-fitting版のfit+transform。

        通常の `fit(X, y).transform(X)` ではなく、scikit-learnの
        `TargetEncoder.fit_transform` に委譲することで、各行が自分自身を含む
        foldの統計量を使わないようにする。
        """
        if y is None:
            raise ValueError("PolarsTargetEncoderのfit_transformにはyが必須です。")
        self.variables_ = as_variable_list(self.variables)
        self.encoder_ = self._build_encoder()
        encoded = self.encoder_.fit_transform(to_numpy_2d(X, self.variables_), np.asarray(y))
        replaced = [pl.Series(col, encoded[:, i]) for i, col in enumerate(self.variables_)]
        return X.with_columns(replaced)


class RareLabelGrouper(BaseEstimator, TransformerMixin):
    """低頻度カテゴリのグループ化（scikit-learnに相当品なし）。

    訓練データにおける出現回数・出現割合が閾値未満のカテゴリを、まとめて
    `other_label` に置き換える。`min_frequency` と `min_ratio` の少なくとも
    一方を指定する必要がある（両方Noneだと「何を低頻度とみなすか」が
    決まらないため）。学習時に存在しなかったカテゴリも `other_label` になる。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        frequent_categories_: 各カラムの「頻出とみなすカテゴリの集合」。
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        min_frequency: int | None = None,
        min_ratio: float | None = None,
        other_label: str = "Other",
    ) -> None:
        self.variables = variables
        self.min_frequency = min_frequency
        self.min_ratio = min_ratio
        self.other_label = other_label

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """訓練データから、頻出とみなすカテゴリを学習する。"""
        if self.min_frequency is None and self.min_ratio is None:
            raise ValueError("min_frequency と min_ratio の少なくとも一方を指定してください。")
        self.variables_ = as_variable_list(self.variables)
        n_rows = X.height
        min_freq = self.min_frequency if self.min_frequency is not None else 0
        min_ratio = self.min_ratio if self.min_ratio is not None else 0.0

        self.frequent_categories_: dict[str, set[Any]] = {}
        for col in self.variables_:
            counts = X[col].value_counts()
            keep = counts.filter(
                (pl.col("count") >= min_freq) & (pl.col("count") / n_rows >= min_ratio)
            )
            self.frequent_categories_[col] = set(keep[col].to_list())
        return self

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """頻出カテゴリ以外を `other_label` に置き換える。"""
        out = X
        for col in self.variables_:
            frequent = self.frequent_categories_[col]
            out = out.with_columns(
                pl.when(pl.col(col).is_in(list(frequent)))
                .then(pl.col(col))
                .otherwise(pl.lit(self.other_label))
                .alias(col)
            )
        return out
