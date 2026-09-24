"""pytest共通設定。"""

import matplotlib
import numpy as np
import polars as pl
import pytest

matplotlib.use("Agg")  # ディスプレイのないテスト環境でも図の保存だけ行う


@pytest.fixture
def synthetic_frame() -> pl.DataFrame:
    """modelingテスト用の合成データ（回帰・二値・多クラスの目的変数とグループ・時刻列を持つ）。"""
    rng = np.random.default_rng(0)
    n = 240
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    return pl.DataFrame(
        {
            "id": np.arange(n),
            "a": a,
            "b": b,
            "cat": rng.choice(["x", "y", "z"], n),
            "g": rng.integers(0, 24, n),
            "ts": pl.datetime_range(
                pl.datetime(2024, 1, 1),
                pl.datetime(2024, 1, 1) + pl.duration(days=n - 1),
                interval="1d",
                eager=True,
            ),
            "y_reg": 2 * a + b + rng.normal(size=n) * 0.1,
            "y_bin": np.where(a + rng.normal(size=n) * 0.3 > 0, "pos", "neg"),
            "y_multi": np.digitize(a, [-0.5, 0.5]),
        }
    )
