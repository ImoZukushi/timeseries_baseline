"""時系列の変換と逆変換（対数・差分・対数差分・季節差分・対数季節差分）。

トレンドや季節性を取り除いた系列でモデルを学習し、予測値を元の尺度に戻すための
scikit-learn互換transformer。対応する変換は次の5種類で、`make_series_transformer` に
名前を渡して作れる。

| 名前 | 変換 | クラス |
|---|---|---|
| `log` | `log(x + offset)` | `feature_engineering.numeric.LogTransformer` |
| `diff` | `x_t - x_{t-1}` | `DifferenceTransformer(periods=1)` |
| `log_diff` | `log x_t - log x_{t-1}` | `DifferenceTransformer(periods=1, log=True)` |
| `seasonal_diff` | `x_t - x_{t-s}` | `DifferenceTransformer(periods=s)` |
| `log_seasonal_diff` | `log x_t - log x_{t-s}` | `DifferenceTransformer(periods=s, log=True)` |

**列の扱い**: 逆変換で元の列に戻せるよう、対象列を同名のまま置き換える
（`feature_engineering.numeric` と同じ方針。`time_series` モジュールのように列を追加はしない）。

**行の並び**: 入力は（系列ごとに）時刻の昇順に並んでいることを前提とする。
`group_by` に系列IDの列を指定すると系列ごとに計算し、系列同士が交互に並んでいてもよい。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Self

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from feature_engineering._polars_sklearn import as_variable_list
from feature_engineering.numeric import LogTransformer

SERIES_TRANSFORM_KINDS = ("log", "diff", "log_diff", "seasonal_diff", "log_seasonal_diff")

# 行の元の並びを保持するための一時列
_ROW = "__row__"

GroupKey = tuple[Any, ...]


class DifferenceTransformer(BaseEstimator, TransformerMixin):
    """差分・対数差分・季節差分・対数季節差分の変換と逆変換。

    変換は `d_t = b_t - b_{t-periods}`（`b = log(x + offset)`（`log=True`）または `x`）。
    各系列の先頭 `periods` 行は、前の値が無いためnullになる。

    逆変換は `b_t = d_t + b_{t-periods}` を累積して元の値に戻す。そのために直前 `periods` 期の
    元の値（起点）が必要で、`fit` 時に各系列の先頭・末尾 `periods` 個の値を記録しておく。
    起点は `inverse_transform` の `history` 引数で次のように選ぶ。

    - `None`（既定）: fitしたデータの先頭の値を起点にし、**fitしたデータと同じ期間** を復元する。
      `sklearn.pipeline.Pipeline.inverse_transform` から呼ばれるのはこの形。
    - `"fit"`: fitしたデータの末尾の値を起点にし、**fitしたデータの直後の期間**
      （予測値など）を復元する。
    - `pl.DataFrame`: 渡した過去データ（元の尺度）の末尾 `periods` 行を起点にし、
      その直後の期間を復元する。

    差分の途中にnull（元データの欠損）があると、それ以降の同じ位相
    （`periods` 期おきの値）は復元できずnullになる。これは差分表現の原理的な限界である。

    Args:
        variables: 対象カラム名。
        periods: 何期前との差を取るか（1なら差分、季節周期なら季節差分）。
        log: Trueなら対数を取ってから差分を取る（対数差分・対数季節差分）。
        offset: 対数を取る前に足す値（`log=True` のときのみ使用）。0を含む系列では1などにする。
        group_by: 系列IDの列（複数系列のパネルデータの場合）。

    Attributes:
        variables_: fitで確定した対象カラム名のリスト。
        group_keys_: 系列IDの列名のリスト（単一系列なら空）。
        head_: 各系列の先頭 `periods` 行の元の値（系列IDの列 + 対象列）。
        tail_: 各系列の末尾 `periods` 行の元の値（系列IDの列 + 対象列）。

    Examples:
        >>> import polars as pl
        >>> X = pl.DataFrame({"x": [1.0, 3.0, 6.0, 10.0]})
        >>> t = DifferenceTransformer("x").fit(X)
        >>> t.transform(X)["x"].to_list()
        [None, 2.0, 3.0, 4.0]
        >>> t.inverse_transform(t.transform(X))["x"].to_list()
        [1.0, 3.0, 6.0, 10.0]
        >>> t.inverse_transform(pl.DataFrame({"x": [5.0, 5.0]}), history="fit")["x"].to_list()
        [15.0, 20.0]
    """

    def __init__(
        self,
        variables: str | Sequence[str],
        periods: int = 1,
        log: bool = False,
        offset: float = 0.0,
        group_by: str | Sequence[str] | None = None,
    ) -> None:
        self.variables = variables
        self.periods = periods
        self.log = log
        self.offset = offset
        self.group_by = group_by

    # --- 学習 -------------------------------------------------------------------------

    def fit(self, X: pl.DataFrame, y: Any = None) -> Self:
        """対象列を確定し、逆変換の起点になる各系列の先頭・末尾の値を記録する。

        Raises:
            ValueError: `periods` が1未満の場合、または `log=True` で
                `x + offset` が0以下の値を含む場合。
        """
        if not isinstance(self.periods, int) or self.periods < 1:
            raise ValueError(f"periods は1以上の整数で指定してください: {self.periods}")
        self.variables_ = as_variable_list(self.variables)
        self.group_keys_ = [] if self.group_by is None else as_variable_list(self.group_by)
        if self.log:
            # 黙ってnullにすると差分・逆変換が壊れるため、学習データでは明示的にエラーにする
            for col in self.variables_:
                n_bad = X.filter(pl.col(col) + self.offset <= 0).height
                if n_bad:
                    raise ValueError(
                        f"{col} に log を取れない値（x + offset <= 0）が {n_bad} 件あります。"
                        " offset を指定してください"
                    )
        columns = [*self.group_keys_, *self.variables_]
        self.head_ = self._per_series(X.select(columns), "head")
        self.tail_ = self._per_series(X.select(columns), "tail")
        return self

    def _per_series(self, frame: pl.DataFrame, which: Literal["head", "tail"]) -> pl.DataFrame:
        """系列ごとの先頭または末尾 `periods` 行を取り出す。"""
        if not self.group_keys_:
            return frame.head(self.periods) if which == "head" else frame.tail(self.periods)
        grouped = frame.group_by(self.group_keys_, maintain_order=True)
        return grouped.head(self.periods) if which == "head" else grouped.tail(self.periods)

    # --- 変換 -------------------------------------------------------------------------

    def _to_base(self, expr: pl.Expr) -> pl.Expr:
        """差分を取る前の値（`log(x + offset)` または `x`）にする。"""
        expr = expr.cast(pl.Float64)
        if not self.log:
            return expr
        shifted = expr + self.offset
        return pl.when(shifted > 0).then(shifted.log()).otherwise(None)

    def transform(self, X: pl.DataFrame) -> pl.DataFrame:
        """差分（`log=True` なら対数差分）を取り、対象列を置き換える。"""
        check_is_fitted(self)
        exprs = []
        for col in self.variables_:
            base = self._to_base(pl.col(col))
            lagged = base.shift(self.periods)
            if self.group_keys_:
                lagged = lagged.over(self.group_keys_)
            exprs.append((base - lagged).alias(col))
        return X.with_columns(exprs)

    # --- 逆変換 ------------------------------------------------------------------------

    def inverse_transform(
        self, X: pl.DataFrame, history: pl.DataFrame | Literal["fit"] | None = None
    ) -> pl.DataFrame:
        """差分系列を元の系列に戻し、対象列を置き換える。

        Args:
            X: 差分系列（`transform` の出力、またはモデルの予測値）。系列ごとに時刻の昇順。
            history: 逆変換の起点。`None` はfitしたデータの先頭（同じ期間を復元）、
                `"fit"` はfitしたデータの末尾（直後の期間を復元）、DataFrameは渡した
                過去データ（元の尺度）の末尾（直後の期間を復元）。

        Returns:
            元の尺度に戻した対象列を持つDataFrame。

        Raises:
            ValueError: 起点が見つからない系列がある、または起点が `periods` 行未満の場合。
        """
        check_is_fitted(self)
        if history is None:
            anchors, continues = self.head_, False
        elif isinstance(history, str):
            if history != "fit":
                raise ValueError(
                    f"history は None / 'fit' / DataFrame で指定してください: {history}"
                )
            anchors, continues = self.tail_, True
        else:
            columns = [*self.group_keys_, *self.variables_]
            anchors, continues = self._per_series(history.select(columns), "tail"), True

        anchor_map = self._anchor_map(anchors)
        restored = {col: np.full(X.height, np.nan) for col in self.variables_}
        for key, rows in self._series_rows(X):
            if key not in anchor_map:
                raise ValueError(f"系列 {key} の逆変換の起点（元の値）がありません")
            for col in self.variables_:
                start = anchor_map[key][col]
                diffs = X[col].cast(pl.Float64).fill_null(np.nan).to_numpy()[rows]
                restored[col][rows] = self._integrate(diffs, start, continues)

        out = []
        for col in self.variables_:
            values = restored[col]
            if self.log:
                values = np.exp(values) - self.offset
            # NaN（復元できない値）はnullにそろえる
            out.append(pl.Series(col, values).fill_nan(None))
        return X.with_columns(out)

    def _anchor_map(self, anchors: pl.DataFrame) -> dict[GroupKey, dict[str, np.ndarray]]:
        """系列ID → {列名: 起点（差分を取る前の尺度）} の辞書を作る。

        Raises:
            ValueError: 起点が `periods` 行未満の系列がある場合。
        """
        base = anchors.with_columns(self._to_base(pl.col(c)).alias(c) for c in self.variables_)
        result: dict[GroupKey, dict[str, np.ndarray]] = {}
        for key, part in self._split(base):
            if part.height < self.periods:
                raise ValueError(
                    f"系列 {key} の起点が {part.height} 行しかありません"
                    f"（periods={self.periods} 行必要です）"
                )
            result[key] = {
                c: part[c].fill_null(np.nan).to_numpy().astype(np.float64) for c in self.variables_
            }
        return result

    def _split(self, frame: pl.DataFrame) -> list[tuple[GroupKey, pl.DataFrame]]:
        """系列ごとに分割する（単一系列なら全体を1つの系列とみなす）。"""
        if not self.group_keys_:
            return [((), frame)]
        parts = frame.partition_by(self.group_keys_, as_dict=True, maintain_order=True)
        return [(tuple(k), v) for k, v in parts.items()]

    def _series_rows(self, X: pl.DataFrame) -> list[tuple[GroupKey, np.ndarray]]:
        """系列ごとに、Xの中での行番号（出現順）を返す。"""
        # 系列IDの列が無い場合に select([]) すると0行になるため、先に行番号を付けてから選ぶ
        indexed = X.with_row_index(_ROW).select(_ROW, *self.group_keys_)
        return [(key, part[_ROW].to_numpy()) for key, part in self._split(indexed)]

    def _integrate(self, diffs: np.ndarray, start: np.ndarray, continues: bool) -> np.ndarray:
        """差分を位相（`t mod periods`）ごとに累積して、差分を取る前の値に戻す。

        Args:
            diffs: 1系列分の差分。
            start: 起点となる `periods` 個の値（差分を取る前の尺度）。
            continues: Trueなら `diffs` は起点の直後から始まる。Falseなら `diffs` の先頭
                `periods` 行は起点そのもの（fitしたデータと同じ期間の復元）。

        Returns:
            差分を取る前の尺度の値。
        """
        p = self.periods
        values = np.full(len(diffs), np.nan)
        for phase in range(p):
            if continues:
                # 直前の値（起点）に、この位相の差分を順に足していく
                values[phase::p] = start[phase] + np.cumsum(diffs[phase::p])
            elif phase < len(diffs):
                # 先頭 periods 行は起点そのもの。それ以降に差分を累積する
                values[phase] = start[phase]
                values[phase + p :: p] = start[phase] + np.cumsum(diffs[phase + p :: p])
        return values


def make_series_transformer(
    kind: str,
    variables: str | Sequence[str],
    seasonal_period: int = 7,
    offset: float = 0.0,
    group_by: str | Sequence[str] | None = None,
) -> LogTransformer | DifferenceTransformer:
    """変換の種類名から、対応するtransformerを作る。

    Args:
        kind: `log` / `diff` / `log_diff` / `seasonal_diff` / `log_seasonal_diff`。
        variables: 対象カラム名。
        seasonal_period: 季節差分の周期（例: 日次データの週周期なら7、年周期なら365）。
        offset: 対数を取る前に足す値（対数系の変換のみ使用）。
        group_by: 系列IDの列（差分系の変換のみ使用。対数は行ごとの変換なので不要）。

    Returns:
        未学習のtransformer。

    Raises:
        ValueError: 未知の種類名の場合。
    """
    if kind == "log":
        return LogTransformer(variables, offset=offset)
    settings: dict[str, tuple[int, bool]] = {
        "diff": (1, False),
        "log_diff": (1, True),
        "seasonal_diff": (seasonal_period, False),
        "log_seasonal_diff": (seasonal_period, True),
    }
    if kind not in settings:
        raise ValueError(f"未知の変換です: {kind}（利用可能: {list(SERIES_TRANSFORM_KINDS)}）")
    periods, log = settings[kind]
    return DifferenceTransformer(
        variables, periods=periods, log=log, offset=offset, group_by=group_by
    )
