"""scripts/train_predict_x50_global.py（グローバルモデル）の入力変換とリーク防止のテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest
import train_predict_x50 as base
import train_predict_x50_global as gm
from test_train_predict_x50 import TARGETS, _frames, _write


def _inputs(tmp_path: Path, **kwargs: object) -> base.Inputs:
    train, test = _frames()
    for column, value in kwargs.items():
        train[column] = value
    return base.load_inputs(*_write(tmp_path, train, test))


def test_common_spec_renames_target_features(tmp_path: Path) -> None:
    spec = gm.common_spec(_inputs(tmp_path))
    assert spec.target == "value"
    assert spec.columns == ["value_lag_1", "value_lag_2", "value_lag_1_ma_3", "value_lag_1_roc_1"]


def test_to_long_has_one_row_per_date_and_series(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    train, test = gm.to_long(inputs)
    assert len(train) == len(inputs.train) * len(TARGETS)
    assert len(test) == len(inputs.test) * len(TARGETS)
    # 日付の昇順、同じ日付の中は目的変数の並び順
    assert train["series"].head(2).tolist() == list(TARGETS)
    assert train["date"].is_monotonic_increasing
    # 縦持ちの目的変数は元の尺度（Yk_raw）
    first = inputs.train.iloc[0]
    assert train.loc[1, "value"] == pytest.approx(first["Y2_raw"])
    # テストに目的変数は無く、元の目的変数の列も持ち込まない
    assert "value" not in test.columns
    assert not set(TARGETS) & set(train.columns)


def test_prepare_rebuilds_features_per_series(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    spec = gm.common_spec(inputs)
    config = gm.make_config("lightgbm", spec, len(TARGETS), cutoffs=["2024-03-01"])
    dataset = gm.prepare(config, inputs, spec)
    assert dataset.frame is not None
    train_idx, valid_idx = dataset.folds[0]
    dates = dataset.frame["date"]
    assert dates.iloc[train_idx].max() < dates.iloc[valid_idx].min()
    # 検証期間には全系列が入る（全系列で同じ境目）
    assert set(dataset.frame["series"].iloc[valid_idx]) == set(TARGETS)
    assert "series" in dataset.X.columns


def test_prepare_rejects_inconsistent_target_features(tmp_path: Path) -> None:
    train, test = _frames()
    # Y2 のラグに Y1 の値が入っている（他の系列が混ざった）CSV
    train["Y2_lag_1"] = train["Y1"].shift(1)
    inputs = base.load_inputs(*_write(tmp_path, train.dropna().reset_index(drop=True), test))
    spec = gm.common_spec(inputs)
    config = gm.make_config("lightgbm", spec, len(TARGETS), cutoffs=["2024-03-01"])
    with pytest.raises(base.LeakageError, match="一致しません"):
        gm.prepare(config, inputs, spec)


def test_common_spec_rejects_different_definitions(tmp_path: Path) -> None:
    train, test = _frames()
    inputs = base.load_inputs(*_write(tmp_path, train.drop(columns="Y2_lag_2"), test))
    with pytest.raises(ValueError, match="定義"):
        gm.common_spec(inputs)
