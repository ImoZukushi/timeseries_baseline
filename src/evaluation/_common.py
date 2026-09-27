"""evaluation パッケージ内で共通に使う小さなヘルパー（公開APIではない）。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def as_1d_float(values: Any) -> np.ndarray:
    """配列・Series・リストを1次元のfloat配列にする。"""
    return np.asarray(values, dtype=np.float64).ravel()


def finite_pair(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    """実測値・予測値のどちらかが欠損（NaN）の行を除いた組を返す。

    Raises:
        ValueError: 長さが違う場合、または有効な行が無い場合。
    """
    t, p = as_1d_float(y_true), as_1d_float(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"y_true と y_pred の長さが違います: {t.shape} != {p.shape}")
    mask = np.isfinite(t) & np.isfinite(p)
    if not mask.any():
        raise ValueError("実測値・予測値がともに有効な行がありません")
    return t[mask], p[mask]


def subsample_index(n: int, max_points: int | None, seed: int = 0) -> np.ndarray:
    """描画用に最大 `max_points` 点を無作為に選んだ行番号（昇順）を返す。"""
    if max_points is None or n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def new_axes(
    ax: Any, ncols: int = 1, nrows: int = 1, figsize: tuple[float, float] = (8, 5)
) -> tuple[plt.Figure, Any]:
    """`ax` が与えられればそのFigureとaxを、無ければ新しいFigureを作って返す。

    複数のaxが必要な部品では、`ax` に `nrows * ncols` 個のaxの並び（配列）を渡せる。
    """
    if ax is None:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, constrained_layout=True, squeeze=False
        )
        return fig, axes if nrows * ncols > 1 else axes[0, 0]
    if nrows * ncols > 1:
        axes = np.asarray(ax, dtype=object)
        if axes.size != nrows * ncols:
            raise ValueError(f"ax は {nrows * ncols} 個必要です（{axes.size} 個が渡されました）")
        return axes.flat[0].figure, axes.reshape(nrows, ncols)
    return ax.figure, ax


def class_labels(n_classes: int, names: Sequence[Any] | None) -> list[str]:
    """クラス番号 0..n-1 の表示名（`names` が無ければ番号の文字列）。"""
    if names is None:
        return [str(k) for k in range(n_classes)]
    return [str(name) for name in names]
