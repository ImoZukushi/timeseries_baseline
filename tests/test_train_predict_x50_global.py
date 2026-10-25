"""scripts/train_predict_x50_global.py（グローバルモデル）の入力とリーク防止のテスト。"""

from __future__ import annotations

from pathlib import Path

import pytest
import train_predict_x50 as base
import train_predict_x50_global as gm
from test_train_predict_x50 import CATEGORIES, _frames, _write


def _inputs(tmp_path: Path) -> base.Inputs:
    return base.load_inputs(*_write(tmp_path, *_frames()))


def test_model_frames_keep_all_categories(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    train, test = gm.model_frames(inputs)
    assert len(train) == len(inputs.train)
    assert len(test) == len(inputs.test)
    # 日付の昇順、同じ日付の中は種類の順
    assert train["Y_cat"].head(2).tolist() == list(CATEGORIES)
    assert train["date"].is_monotonic_increasing
    # 目的変数は元の尺度（Y_raw）。CSVの目的変数の特徴量は外す
    assert train["Y"].tolist() == pytest.approx(inputs.train["Y_raw"].tolist())
    assert not [c for c in train.columns if c.startswith("Y_lag")]
    # テストに目的変数は無い
    assert "Y" not in test.columns


def test_prepare_rebuilds_features_per_series(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    config = gm.make_config("lightgbm", inputs.spec, len(CATEGORIES), cutoffs=["2024-03-01"])
    dataset = gm.prepare(config, inputs)
    assert dataset.frame is not None
    train_idx, valid_idx = dataset.folds[0]
    dates = dataset.frame["date"]
    assert dates.iloc[train_idx].max() < dates.iloc[valid_idx].min()
    # 検証期間には全系列が入る（全系列で同じ境目）
    assert set(dataset.frame["Y_cat"].iloc[valid_idx]) == set(CATEGORIES)
    assert "Y_cat" in dataset.X.columns
    assert not {"Y", "Y_raw"} & set(dataset.X.columns)


def test_prepare_rejects_features_mixed_across_series(tmp_path: Path) -> None:
    train, test = _frames()
    # Y2 の行のラグに Y1 の値が入っている（他の系列が混ざった）CSV
    is_y1, is_y2 = train["Y_cat"] == "Y1", train["Y_cat"] == "Y2"
    train.loc[is_y2, "Y_lag_1"] = train.loc[is_y1, "Y_lag_1"].to_numpy()
    inputs = base.load_inputs(*_write(tmp_path, train, test))
    config = gm.make_config("lightgbm", inputs.spec, len(CATEGORIES), cutoffs=["2024-03-01"])
    with pytest.raises(base.LeakageError, match="一致しません"):
        gm.prepare(config, inputs)


def test_make_config_uses_category_as_series() -> None:
    spec = base.parse_target_features(["Y_lag_1"])
    config = gm.make_config("lightgbm", spec, 3)
    assert config.forecast is not None and config.forecast.series_col == "Y_cat"
    assert config.data.target == "Y"
    with pytest.raises(ValueError, match="モデル"):
        gm.make_config("random_forest", spec, 3)
