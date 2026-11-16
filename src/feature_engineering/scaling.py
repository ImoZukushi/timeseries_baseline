"""スケーリング（正規化）の特徴量エンジニアリング（scikit-learn互換transformer）。

各クラスは `sklearn.base.BaseEstimator` / `TransformerMixin` を継承する。
本モジュールの変換はいずれも対象列を同名のまま置き換え、全て `inverse_transform`
（予測値等を元のスケールに戻す用途）を実装する。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.utils.validation import check_is_fitted

from feature_engineering._pandas_sklearn import as_variable_list, replace_columns, to_numpy_2d


class PandasMinMaxScaler(BaseEstimator, TransformerMixin):
    """最大値・最小値によるスケーリング（`sklearn.preprocessing.MinMaxScaler` をラップ）。

    `(x - min) / (max - min)` で[0, 1]区間に変換する。訓練データが定数列
    （max == min）の場合はscikit-learn側の仕様により0を返す（0除算にはならない）。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        scaler_: fit済みの内部 `MinMaxScaler`。
    """

    def __init__(self, variables: str | Sequence[str]) -> None:
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """訓練データから最大値・最小値を学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.scaler_ = MinMaxScaler()
        self.scaler_.fit(to_numpy_2d(X, self.variables_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """学習済みの最大値・最小値でスケーリングする。"""
        scaled = self.scaler_.transform(to_numpy_2d(X, self.variables_))
        return replace_columns(X, {col: scaled[:, i] for i, col in enumerate(self.variables_)})

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """元のスケールに戻す。"""
        original = self.scaler_.inverse_transform(to_numpy_2d(X, self.variables_))
        return replace_columns(X, {col: original[:, i] for i, col in enumerate(self.variables_)})


class PandasRobustScaler(BaseEstimator, TransformerMixin):
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

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """訓練データから中央値・分位数を学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.scaler_ = RobustScaler(quantile_range=self.quantile_range)
        self.scaler_.fit(to_numpy_2d(X, self.variables_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """学習済みの中央値・分位数でスケーリングする。"""
        scaled = self.scaler_.transform(to_numpy_2d(X, self.variables_))
        return replace_columns(X, {col: scaled[:, i] for i, col in enumerate(self.variables_)})

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """元のスケールに戻す。"""
        original = self.scaler_.inverse_transform(to_numpy_2d(X, self.variables_))
        return replace_columns(X, {col: original[:, i] for i, col in enumerate(self.variables_)})


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

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """訓練データから平均値・最小値・最大値を学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.mean_: dict[str, float] = {col: float(X[col].mean()) for col in self.variables_}
        self.min_: dict[str, float] = {col: float(X[col].min()) for col in self.variables_}
        self.max_: dict[str, float] = {col: float(X[col].max()) for col in self.variables_}
        return self

    def _scale(self, col: str) -> float:
        scale = self.max_[col] - self.min_[col]
        return scale if scale != 0 else 1.0

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """学習済みの平均値・値域で平均正規化する。"""
        return replace_columns(
            X,
            {
                col: (X[col].astype(float) - self.mean_[col]) / self._scale(col)
                for col in self.variables_
            },
        )

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """元のスケールに戻す。"""
        return replace_columns(
            X,
            {
                col: X[col].astype(float) * self._scale(col) + self.mean_[col]
                for col in self.variables_
            },
        )


class PandasMaxAbsScaler(BaseEstimator, TransformerMixin):
    """最大絶対値スケーリング（`sklearn.preprocessing.MaxAbsScaler` をラップ）。

    `x / max(|x|)` で[-1, 1]区間に変換する。訓練データが全て0の場合は
    scikit-learn側の仕様によりそのまま0を返す（0除算にはならない）。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        scaler_: fit済みの内部 `MaxAbsScaler`。
    """

    def __init__(self, variables: str | Sequence[str]) -> None:
        self.variables = variables

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """訓練データから最大絶対値を学習する。"""
        self.variables_ = as_variable_list(self.variables)
        self.scaler_ = MaxAbsScaler()
        self.scaler_.fit(to_numpy_2d(X, self.variables_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """学習済みの最大絶対値でスケーリングする。"""
        scaled = self.scaler_.transform(to_numpy_2d(X, self.variables_))
        return replace_columns(X, {col: scaled[:, i] for i, col in enumerate(self.variables_)})

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """元のスケールに戻す。"""
        original = self.scaler_.inverse_transform(to_numpy_2d(X, self.variables_))
        return replace_columns(X, {col: original[:, i] for i, col in enumerate(self.variables_)})


GROUP_SCALING_METHODS = ("standardize", "normalize")


class GroupScaler(BaseEstimator, TransformerMixin):
    """グループ（系列IDなど）ごとの標準化・正規化。

    - `"standardize"`（標準化）: `(x - 平均) / 標準偏差`（標準偏差は ddof=0）。
    - `"normalize"`（正規化）: `(x - 最小値) / (最大値 - 最小値)`。

    平均・標準偏差・最小値・最大値は、グループごとに、fit に渡した行だけから求める
    （CV では各 fold の学習側）。統計量は欠損・±inf を除いて計算する。

    - ばらつきが無いグループ（標準偏差 0、または最大値 = 最小値）は、0除算を避けるため
      分母を1にする。
    - 学習に無いグループ・グループ列が欠損の行・学習側の値がすべて欠損のグループの列は、
      fit に渡した全行の統計量を使う（全行でも値が無い列は、中心0・分母1 = 変換しない）。
    - 欠損は欠損のまま、±inf は符号を保ったまま返す。
    - グループの列そのものは変換しない（後段のエンコーダで使うため、そのまま残す）。

    Args:
        variables: 対象カラム名、またはその並び。
        group_col: グループの列名。
        method: `"standardize"` または `"normalize"`。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        center_: グループ × 対象カラムの中心（標準化は平均、正規化は最小値）。
        scale_: グループ × 対象カラムの分母（標準化は標準偏差、正規化は最大値 − 最小値）。
        global_center_: 全行の中心（学習に無いグループに使う）。
        global_scale_: 全行の分母（学習に無いグループに使う）。

    Raises:
        ValueError: `method` が未対応、または `group_col` を対象カラムに含めた場合。
        KeyError: 対象カラム・グループの列が無い場合。
    """

    def __init__(
        self, variables: str | Sequence[str], group_col: str, method: str = "standardize"
    ) -> None:
        self.variables = variables
        self.group_col = group_col
        self.method = method

    def fit(self, X: pd.DataFrame, y: Any = None) -> Self:
        """グループごとの中心と分母を学習する。"""
        if self.method not in GROUP_SCALING_METHODS:
            raise ValueError(
                f"method は {GROUP_SCALING_METHODS} のどれかにしてください: {self.method}"
            )
        self.variables_ = as_variable_list(self.variables)
        if self.group_col in self.variables_:
            raise ValueError(f"グループの列はスケーリングできません: {self.group_col}")
        missing = [c for c in [self.group_col, *self.variables_] if c not in X.columns]
        if missing:
            raise KeyError(f"列がありません: {missing}")
        # ±inf は統計量に入れない（平均・最大値が inf になるのを防ぐ）
        values = self._values(X).replace([np.inf, -np.inf], np.nan)
        grouped = values.groupby(X[self.group_col].to_numpy(), sort=False)
        if self.method == "standardize":
            center, scale = grouped.mean(), grouped.std(ddof=0)
            global_center, global_scale = values.mean(), values.std(ddof=0)
        else:
            center, scale = grouped.min(), grouped.max() - grouped.min()
            global_center, global_scale = values.min(), values.max() - values.min()
        # 全行でも値が無い列は変換しない（中心0・分母1）。分母が0の列は分母を1にする
        self.global_center_ = global_center.fillna(0.0)
        self.global_scale_ = self._safe_scale(global_scale)
        # グループで値が無い列は、全行の統計量で補う
        self.center_ = center.fillna(self.global_center_)
        self.scale_ = self._safe_scale(scale.where(center.notna(), self.global_scale_, axis=1))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """グループごとの中心と分母でスケーリングする。"""
        check_is_fitted(self, "center_")
        center, scale = self._row_statistics(X)
        scaled = (self._values(X).to_numpy() - center) / scale
        return replace_columns(X, {col: scaled[:, i] for i, col in enumerate(self.variables_)})

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """元のスケールに戻す（グループの列が必要）。"""
        check_is_fitted(self, "center_")
        center, scale = self._row_statistics(X)
        original = self._values(X).to_numpy() * scale + center
        return replace_columns(X, {col: original[:, i] for i, col in enumerate(self.variables_)})

    def _values(self, X: pd.DataFrame) -> pd.DataFrame:
        """対象カラムを浮動小数の DataFrame にする（nullable 型の欠損は NaN にする）。"""
        return X[self.variables_].astype("float64")

    def _row_statistics(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """行ごとの (中心, 分母)。学習に無いグループの行は全行の統計量を使う。"""
        groups = X[self.group_col].to_numpy()
        center = self.center_.reindex(groups).fillna(self.global_center_)
        scale = self.scale_.reindex(groups).fillna(self.global_scale_)
        return center.to_numpy(dtype=np.float64), scale.to_numpy(dtype=np.float64)

    @staticmethod
    def _safe_scale(scale: Any) -> Any:
        """分母（Series または DataFrame）が 0 または欠損なら1にする。"""
        return scale.where(scale > 0, 1.0)
