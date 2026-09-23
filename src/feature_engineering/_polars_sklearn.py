"""polarsとscikit-learn transformerの橋渡しをする内部ヘルパー。

本パッケージ内のtransformerクラスから共通で使う、ごく小さいユーティリティのみを置く。
公開APIではないため、このモジュールを直接importすることは想定していない。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl


def as_variable_list(variables: str | Sequence[str]) -> list[str]:
    """`variables` 引数（単一カラム名または複数カラム名）を常にリスト形式に正規化する。

    Args:
        variables: 対象カラム名、またはその並び。

    Returns:
        カラム名のリスト。
    """
    if isinstance(variables, str):
        return [variables]
    return list(variables)


def to_numpy_2d(X: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """polars DataFrameの指定カラムを (n_samples, n_columns) のnumpy配列に変換する。

    Args:
        X: 対象のDataFrame。
        columns: 変換対象のカラム名リスト。

    Returns:
        指定カラムからなるnumpy配列。
    """
    return X.select(columns).to_numpy()
