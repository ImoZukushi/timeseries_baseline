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
    irregular = gen._irregular_masks(len(frame), 0)
    for col in gen.TARGET_COLS:
        # 欠損を除いた観測値で、全期間・学習・テストのどれも棄却されない
        # （irregular_flg が True の行は使わないので除く）
        used = frame[col].mask(irregular[col])
        for part in (used, used[~is_test], used[is_test]):
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


def test_split_adds_irregular_flag_and_no_answer() -> None:
    _, frame = _frames()
    parts = gen.split_train_test(frame, TEST_START, 0)
    # 学習・テストの2つだけで、テストの正解は作らない
    assert len(parts) == 2
    train, test = parts
    assert "Y" in train.columns and "Y" not in test.columns
    for part in (train, test):
        assert part[gen.IRREGULAR_COL].dtype == bool
        assert 0.01 < part[gen.IRREGULAR_COL].mean() < 0.05
    # 種類・日付ごとのフラグの位置が、検定から除いた行と一致する
    masks = gen._irregular_masks(len(frame), 0)
    y1 = train[train["Y_cat"] == "Y1"]
    assert y1[gen.IRREGULAR_COL].tolist() == masks["Y1"][: len(y1)].tolist()
