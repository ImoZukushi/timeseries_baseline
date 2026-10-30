"""scripts/generate_dummy_x50.py（ダミーデータの生成）のテスト。"""

from __future__ import annotations

import generate_dummy_x50 as gen
import numpy as np
import pandas as pd

TEST_START = "2025-01-01"


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    full = gen.make_dummy_frame("2022-01-01", 1461, 0, TEST_START)
    return full, gen.add_missing(full, TEST_START, 0)


def test_targets_have_no_autocorrelation_at_alpha_010() -> None:
    _, frame = _frames()
    is_test = frame["date"] >= pd.Timestamp(TEST_START)
    for col in gen.TARGET_COLS:
        # 欠損を除いた観測値で、全期間・学習・テストのどれも棄却されない
        for part in (frame[col], frame.loc[~is_test, col], frame.loc[is_test, col]):
            assert gen.ljung_box_min_pvalue(part.to_numpy()) > gen.LJUNG_BOX_ALPHA, col


def test_targets_still_depend_on_features_and_have_missing_values() -> None:
    full, frame = _frames()
    # 説明変数の効果は残っている（X11 は Y1 に効く）
    assert abs(np.corrcoef(np.log(full["X11"]), full["Y1"])[0, 1]) > 0.1
    # 欠損が入り、欠損でない値は変わらない
    assert frame["Y1"].isna().any()
    observed = frame["Y1"].notna()
    assert frame.loc[observed, "Y1"].equals(full.loc[observed, "Y1"])
    # 説明変数の季節性・トレンドは残る（X1〜X10 は自己相関を持つ）
    assert gen.ljung_box_min_pvalue(full["X1"].to_numpy()) < gen.LJUNG_BOX_ALPHA
