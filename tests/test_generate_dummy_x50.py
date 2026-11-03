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


def test_split_adds_int_flags_and_no_answer() -> None:
    _, frame = _frames()
    parts = gen.split_train_test(frame, TEST_START, 0)
    # 学習・テストの2つだけで、テストの正解は作らない
    assert len(parts) == 2
    train, test = parts
    assert "Y" in train.columns and "Y" not in test.columns
    # irregular_flg は学習・テストの両方、eliminate_flg は学習だけ。どれも int 型の 0/1
    assert gen.ELIMINATE_COL in train.columns and gen.ELIMINATE_COL not in test.columns
    for part, cols in (
        (train, [gen.IRREGULAR_COL, gen.ELIMINATE_COL]),
        (test, [gen.IRREGULAR_COL]),
    ):
        for col in cols:
            assert part[col].dtype == "int64" and set(part[col].unique()) == {0, 1}, col
            assert 0.01 < part[col].mean() < 0.05, col
    # 種類・日付ごとの irregular_flg の位置が、検定から除いた行と一致する
    masks = gen._irregular_masks(len(frame), 0)
    y1 = train[train["Y_cat"] == "Y1"]
    assert y1[gen.IRREGULAR_COL].tolist() == masks["Y1"][: len(y1)].astype(int).tolist()


def test_irregular_rows_have_out_of_range_targets() -> None:
    _, frame = _frames()
    train, _ = gen.split_train_test(frame, TEST_START, 0)
    observed = train["Y"].notna()
    irregular = train[gen.IRREGULAR_COL] == 1
    # irregular_flg が 1 の行は、予測の下限 0 より下の値。それ以外は正の値
    assert (train.loc[observed & irregular, "Y"] < 0).all()
    assert (train.loc[observed & ~irregular, "Y"] > 0).all()
    # eliminate_flg の行の Y は変えない（欠損前の値と同じ）
    eliminated = (train[gen.ELIMINATE_COL] == 1) & observed & ~irregular
    original = gen.to_long(frame)
    merged = train[eliminated].merge(original, on=["date", "Y_cat"], suffixes=("", "_orig"))
    assert np.allclose(merged["Y"], merged["Y_orig"])
