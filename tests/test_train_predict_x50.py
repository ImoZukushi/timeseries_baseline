"""scripts/train_predict_x50.py の入力確認（リーク防止）のテスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import train_predict_x50 as tp

TARGETS = ("Y1", "Y2")


def _frames(n_train: int = 90, n_test: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """003 のノートブックの出力と同じ形の、小さな学習・テストデータを作る（目的変数2つ）。"""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n_train + n_test, freq="D")
    columns: dict[str, object] = {"date": dates}
    features: dict[str, object] = {}
    for k, target in enumerate(TARGETS):
        y_raw = 50 * (k + 1) + np.cumsum(rng.normal(size=n_train + n_test))
        y = pd.Series(np.log(y_raw))
        lag1 = y.shift(1)
        columns[target] = y
        columns[f"{target}_raw"] = y_raw
        # 過去の値だけから作った目的変数の特徴量（modeling.forecasting と同じ定義）
        features |= {
            f"{target}_lag_1": lag1,
            f"{target}_lag_2": y.shift(2),
            f"{target}_lag_1_ma_3": lag1.rolling(3).mean(),
            f"{target}_lag_1_roc_1": (lag1 - lag1.shift(1)) / lag1.shift(1),
        }
    full = pd.DataFrame(
        {
            **columns,
            **features,
            "date_month": dates.month,
            "X1": rng.normal(size=n_train + n_test),
        }
    )
    train = full.iloc[:n_train].dropna().reset_index(drop=True)
    test = full.iloc[n_train:][["date", "date_month", "X1"]].reset_index(drop=True)
    return train, test


def _write(tmp_path: Path, train: pd.DataFrame, test: pd.DataFrame) -> tuple[Path, Path]:
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    train.to_csv(train_path, index=False, date_format="%Y-%m-%d")
    test.to_csv(test_path, index=False, date_format="%Y-%m-%d")
    return train_path, test_path


def test_detect_targets_and_parse_features() -> None:
    columns = ["date", "Y1", "Y1_raw", "Y10", "Y10_raw", "Y1_lag_1", "Y1_lag_7", "Y1_lag_1_ma_7"]
    columns += ["Y1_lag_1_roc_1", "Y10_lag_1", "X1"]
    assert tp.detect_targets(columns) == ["Y1", "Y10"]
    spec = tp.parse_target_features(columns, "Y1")
    assert (spec.lags, spec.rolling_windows, spec.rate_of_change) == ([1, 7], [7], True)
    # Y10 の列は Y1 の特徴量に含めない
    assert spec.columns == ["Y1_lag_1", "Y1_lag_7", "Y1_lag_1_ma_7", "Y1_lag_1_roc_1"]
    assert tp.parse_target_features(columns, "Y10").columns == ["Y10_lag_1"]
    # 再帰予測で作り直せない特徴量（2日前のラグの移動平均）はエラー
    with pytest.raises(ValueError, match="作り直せない"):
        tp.parse_target_features(["Y1_lag_2", "Y1_lag_2_ma_7"], "Y1")


def test_load_inputs_accepts_valid_files(tmp_path: Path) -> None:
    inputs = tp.load_inputs(*_write(tmp_path, *_frames()))
    assert inputs.targets == list(TARGETS)
    assert inputs.exog_cols == ["date_month", "X1"]
    assert inputs.specs["Y2"].lags == [1, 2]
    assert not set(TARGETS) & set(inputs.test.columns)
    # 一部の目的変数だけを選べる
    assert tp.load_inputs(*_write(tmp_path, *_frames()), targets=["Y2"]).targets == ["Y2"]
    with pytest.raises(ValueError, match="無い目的変数"):
        tp.load_inputs(*_write(tmp_path, *_frames()), targets=["Y9"])


@pytest.mark.parametrize("leaked", ["Y1", "Y2_raw", "Y2_lag_1"])
def test_load_inputs_rejects_target_in_test(tmp_path: Path, leaked: str) -> None:
    train, test = _frames()
    test[leaked] = 1.0
    with pytest.raises(tp.LeakageError):
        tp.load_inputs(*_write(tmp_path, train, test))


def test_load_inputs_rejects_mismatched_columns_and_gaps(tmp_path: Path) -> None:
    train, test = _frames()
    with pytest.raises(ValueError, match="特徴量の列が違います"):
        tp.load_inputs(*_write(tmp_path, train, test.drop(columns="X1")))
    # テストデータが学習データの翌日から始まっていない
    with pytest.raises(ValueError, match="翌日"):
        tp.load_inputs(*_write(tmp_path, train, test.iloc[1:]))


@pytest.mark.parametrize("target", TARGETS)
def test_prepare_uses_only_own_target_features(tmp_path: Path, target: str) -> None:
    inputs = tp.load_inputs(*_write(tmp_path, *_frames()))
    config = tp.make_config("lightgbm", inputs.specs[target], cutoffs=["2024-03-01"])
    dataset = tp.prepare(config, inputs)
    other = next(t for t in TARGETS if t != target)
    assert not [c for c in dataset.X.columns if c == other or c.startswith(f"{other}_")]
    assert inputs.specs[target].columns == [
        c for c in dataset.X.columns if c.startswith(f"{target}_")
    ]
    train_idx, valid_idx = dataset.folds[0]
    assert train_idx.max() < valid_idx.min()
    assert dataset.test_frame is not None and target not in dataset.test_frame.columns


def test_prepare_rejects_target_features_with_current_value(tmp_path: Path) -> None:
    train, test = _frames()
    # ラグのつもりで当日の値が入っている（リーク）CSV
    train["Y2_lag_1"] = train["Y2"]
    inputs = tp.load_inputs(*_write(tmp_path, train, test))
    config = tp.make_config("lightgbm", inputs.specs["Y2"], cutoffs=["2024-03-01"])
    with pytest.raises(tp.LeakageError, match="一致しません"):
        tp.prepare(config, inputs)
    # Y1 の特徴量は正しいので、Y1 は通る
    tp.prepare(tp.make_config("lightgbm", inputs.specs["Y1"], cutoffs=["2024-03-01"]), inputs)


def test_make_config_rejects_unknown_model() -> None:
    spec = tp.parse_target_features(["Y1_lag_1"], "Y1")
    with pytest.raises(ValueError, match="モデル"):
        tp.make_config("random_forest", spec)
    assert tp.make_config("xgboost", spec).name == "x50_xgboost_Y1"
