"""テストで pandas の結果を比べやすくする小さなヘルパー。

pandas では欠損が NaN / NaT / pd.NA になるため、期待値を `None` で書けるよう、
リストや辞書に変換するときに欠損を `None` にそろえる。
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _none_if_missing(value: Any) -> Any:
    """欠損（NaN・NaT・pd.NA）なら None、それ以外はそのまま返す。"""
    try:
        return None if pd.isna(value) else value
    except (TypeError, ValueError):
        return value  # リストなど、欠損の判定ができない値


def as_list(values: pd.Series | pd.Index | list[Any]) -> list[Any]:
    """Series をリストにする（欠損は None。`series.pipe(as_list)` の形でも使える）。"""
    return [_none_if_missing(v) for v in list(values)]


def row_at(frame: pd.DataFrame, i: int) -> dict[str, Any]:
    """i 番目（0始まりの位置）の行を、列名 → 値 の辞書にする（欠損は None）。"""
    return {col: _none_if_missing(v) for col, v in frame.iloc[i].to_dict().items()}


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """全行を辞書のリストにする（欠損は None）。"""
    return [row_at(frame, i) for i in range(len(frame))]
