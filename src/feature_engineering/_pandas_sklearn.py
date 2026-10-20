"""pandasとscikit-learn transformerの橋渡しをする内部ヘルパー。

本パッケージ内のtransformerクラスから共通で使う、ごく小さいユーティリティのみを置く。
公開APIではないため、このモジュールを直接importすることは想定していない。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


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


def to_numpy_2d(X: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """pandas DataFrameの指定カラムを (n_samples, n_columns) のnumpy配列に変換する。

    Args:
        X: 対象のDataFrame。
        columns: 変換対象のカラム名リスト。

    Returns:
        指定カラムからなるnumpy配列。
    """
    return X[columns].to_numpy()


def replace_columns(X: pd.DataFrame, values: Mapping[str, np.ndarray | pd.Series]) -> pd.DataFrame:
    """指定した列を置き換えた（または追加した）新しい DataFrame を返す（元の X は変えない）。

    numpy 配列は X の行の並びに合わせて置き、行のインデックスは X のまま保つ。

    Args:
        X: 元の DataFrame。
        values: 列名 → 新しい値（numpy 配列、または X と同じインデックスの Series）。

    Returns:
        列を置き換えた DataFrame。
    """
    out = X.copy()
    for col, value in values.items():
        out[col] = pd.Series(value, index=X.index) if isinstance(value, np.ndarray) else value
    return out
