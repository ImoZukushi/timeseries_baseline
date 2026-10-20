"""OOF予測・テスト予測の保存と読み込み。

アンサンブルはモデルを再学習せず、各実験で保存した予測ファイルを入力にする。
予測ファイルは次の列を持つParquet:

- `row`: 学習/テストデータ内の行番号（ソート後の順序）
- `id`: `data.id_col` の値（指定時のみ）
- `fold`: 検証データになったfold番号（OOFのみ。どのfoldにも入らない行は-1）
- `target`: エンコード済みの目的変数（OOFのみ）
- `pred`（二値・回帰）または `pred_0`, `pred_1`, ...（多クラス）: 予測値
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from util.paths import ensure_parent_dir

OOF_FILENAME = "oof_predictions.parquet"
TEST_FILENAME = "test_predictions.parquet"


def predictions_to_frame(
    pred: np.ndarray,
    ids: pd.Series | None = None,
    fold_ids: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> pd.DataFrame:
    """予測値の配列を保存用のDataFrameに変換する。

    Args:
        pred: 予測値。(n,) または (n, n_classes)。
        ids: 行IDの列。
        fold_ids: fold番号。
        target: 目的変数。

    Returns:
        保存用のDataFrame。
    """
    columns: dict[str, Any] = {"row": np.arange(len(pred), dtype=np.int64)}
    if ids is not None:
        # 元のインデックスとは揃えず、並び順のまま使う
        columns["id"] = ids.reset_index(drop=True)
    if fold_ids is not None:
        columns["fold"] = fold_ids
    if target is not None:
        columns["target"] = target
    if pred.ndim == 2:
        for k in range(pred.shape[1]):
            columns[f"pred_{k}"] = pred[:, k]
    else:
        columns["pred"] = pred
    # どのfoldにも入らなかった行の予測はNaN（Parquetにはnullとして書かれる）
    return pd.DataFrame(columns)


def prediction_columns(frame: pd.DataFrame) -> list[str]:
    """予測値の列名（`pred` または `pred_0`, `pred_1`, ...）を返す。"""
    if "pred" in frame.columns:
        return ["pred"]
    return sorted(
        (c for c in frame.columns if c.startswith("pred_")), key=lambda c: int(c.split("_")[1])
    )


def frame_to_predictions(frame: pd.DataFrame) -> np.ndarray:
    """保存した予測DataFrameから予測値の配列を取り出す（欠損はNaN）。"""
    cols = prediction_columns(frame)
    values = frame[cols].to_numpy(dtype=np.float64, na_value=np.nan)
    return values[:, 0] if cols == ["pred"] else values


def save_predictions(frame: pd.DataFrame, path: Path) -> Path:
    """予測DataFrameをParquetで保存する（インデックスは保存しない）。"""
    ensure_parent_dir(path)
    frame.to_parquet(path, index=False)
    return path
