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

import numpy as np
import polars as pl

from util.paths import ensure_parent_dir

OOF_FILENAME = "oof_predictions.parquet"
TEST_FILENAME = "test_predictions.parquet"


def predictions_to_frame(
    pred: np.ndarray,
    ids: pl.Series | None = None,
    fold_ids: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> pl.DataFrame:
    """予測値の配列を保存用のDataFrameに変換する。

    Args:
        pred: 予測値。(n,) または (n, n_classes)。
        ids: 行IDの列。
        fold_ids: fold番号。
        target: 目的変数。

    Returns:
        保存用のDataFrame。
    """
    columns: dict[str, pl.Series] = {"row": pl.Series("row", np.arange(len(pred)))}
    if ids is not None:
        columns["id"] = ids.alias("id")
    if fold_ids is not None:
        columns["fold"] = pl.Series("fold", fold_ids)
    if target is not None:
        columns["target"] = pl.Series("target", target)
    if pred.ndim == 2:
        for k in range(pred.shape[1]):
            columns[f"pred_{k}"] = pl.Series(f"pred_{k}", pred[:, k])
    else:
        columns["pred"] = pl.Series("pred", pred)
    # NaNはnullに揃えておく（どのfoldにも入らなかった行）
    return pl.DataFrame(list(columns.values())).with_columns(pl.selectors.float().fill_nan(None))


def prediction_columns(frame: pl.DataFrame) -> list[str]:
    """予測値の列名（`pred` または `pred_0`, `pred_1`, ...）を返す。"""
    if "pred" in frame.columns:
        return ["pred"]
    return sorted(
        (c for c in frame.columns if c.startswith("pred_")), key=lambda c: int(c.split("_")[1])
    )


def frame_to_predictions(frame: pl.DataFrame) -> np.ndarray:
    """保存した予測DataFrameから予測値の配列を取り出す（nullはNaN）。"""
    cols = prediction_columns(frame)
    values = frame.select(cols).to_numpy().astype(np.float64)
    return values[:, 0] if cols == ["pred"] else values


def save_predictions(frame: pl.DataFrame, path: Path) -> Path:
    """予測DataFrameをParquetで保存する。"""
    ensure_parent_dir(path)
    frame.write_parquet(path)
    return path
