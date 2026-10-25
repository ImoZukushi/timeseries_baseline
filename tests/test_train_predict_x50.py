"""scripts/train_predict_x50.py の入力確認（リーク防止）のテスト（long 型の入力）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import train_predict_x50 as tp

CATEGORIES = ("Y1", "Y2")


def _frames(n_train: int = 90, n_test: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """003 のノートブックの出力と同じ形（long 型）の、小さな学習・テストデータを作る（種類2つ）。"""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n_train + n_test, freq="D")
    x1 = rng.normal(size=n_train + n_test)
    parts = []
    for k, category in enumerate(CATEGORIES):
        y_raw = 50 * (k + 1) + np.cumsum(rng.normal(size=n_train + n_test))
        y = pd.Series(np.log(y_raw))
        lag1 = y.shift(1)
        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "Y_cat": category,
                    "Y": y,
                    "Y_raw": y_raw,
                    # 種類ごとに、過去の値だけから作った目的変数の特徴量
                    # （modeling.forecasting と同じ定義）
                    "Y_lag_1": lag1,
                    "Y_lag_2": y.shift(2),
                    "Y_lag_1_ma_3": lag1.rolling(3).mean(),
                    "Y_lag_1_roc_1": (lag1 - lag1.shift(1)) / lag1.shift(1),
                    "date_month": dates.month,
                    "X1": x1,
                }
            )
        )
    full = tp.sort_long(pd.concat(parts, ignore_index=True), list(CATEGORIES))
    is_train = full["date"] < dates[n_train]
    train = full[is_train].dropna().reset_index(drop=True)
    test = full.loc[~is_train, ["date", "Y_cat", "date_month", "X1"]].reset_index(drop=True)
    return train, test


def _write(tmp_path: Path, train: pd.DataFrame, test: pd.DataFrame) -> tuple[Path, Path]:
    train_path, test_path = tmp_path / "train.csv", tmp_path / "test.csv"
    train.to_csv(train_path, index=False, date_format="%Y-%m-%d")
    test.to_csv(test_path, index=False, date_format="%Y-%m-%d")
    return train_path, test_path


def test_category_order_and_parse_features() -> None:
    frame = pd.DataFrame({"Y_cat": ["Y1", "Y10", "Y2", "Y1"]})
    assert tp.category_order(frame) == ["Y1", "Y10", "Y2"]
    columns = ["date", "Y_cat", "Y", "Y_raw", "Y_lag_1", "Y_lag_7", "Y_lag_1_ma_7", "Y_lag_1_roc_1"]
    spec = tp.parse_target_features([*columns, "X1"])
    assert (spec.lags, spec.rolling_windows, spec.rate_of_change) == ([1, 7], [7], True)
    assert spec.columns == ["Y_lag_1", "Y_lag_7", "Y_lag_1_ma_7", "Y_lag_1_roc_1"]
    # 再帰予測で作り直せない特徴量（2日前のラグの移動平均）はエラー
    with pytest.raises(ValueError, match="作り直せない"):
        tp.parse_target_features(["Y_lag_2", "Y_lag_2_ma_7"])


def test_load_inputs_accepts_valid_files(tmp_path: Path) -> None:
    inputs = tp.load_inputs(*_write(tmp_path, *_frames()))
    assert inputs.categories == list(CATEGORIES)
    assert inputs.exog_cols == ["date_month", "X1"]
    assert inputs.spec.lags == [1, 2]
    assert "Y" not in inputs.test.columns
    # 日付の昇順、同じ日付の中は種類の順
    assert inputs.train["Y_cat"].head(2).tolist() == list(CATEGORIES)
    # 一部の種類だけを選べる
    only_y2 = tp.load_inputs(*_write(tmp_path, *_frames()), categories=["Y2"])
    assert only_y2.categories == ["Y2"] and set(only_y2.train["Y_cat"]) == {"Y2"}
    with pytest.raises(ValueError, match="無い種類"):
        tp.load_inputs(*_write(tmp_path, *_frames()), categories=["Y9"])


@pytest.mark.parametrize("leaked", ["Y", "Y_raw", "Y_lag_1"])
def test_load_inputs_rejects_target_in_test(tmp_path: Path, leaked: str) -> None:
    train, test = _frames()
    test[leaked] = 1.0
    with pytest.raises(tp.LeakageError):
        tp.load_inputs(*_write(tmp_path, train, test))


def test_load_inputs_rejects_mismatched_columns_and_gaps(tmp_path: Path) -> None:
    train, test = _frames()
    with pytest.raises(ValueError, match="特徴量の列が違います"):
        tp.load_inputs(*_write(tmp_path, train, test.drop(columns="X1")))
    # テストデータが学習データの翌日から始まっていない（先頭の日を全種類で除く）
    with pytest.raises(ValueError, match="翌日"):
        tp.load_inputs(*_write(tmp_path, train, test[test["date"] > test["date"].min()]))
    # ある種類の日付に欠けがある
    gap = train.drop(index=train.index[(train["Y_cat"] == "Y2")][10])
    with pytest.raises(ValueError, match="1日刻み"):
        tp.load_inputs(*_write(tmp_path, gap, test))


@pytest.mark.parametrize("category", CATEGORIES)
def test_prepare_uses_only_rows_of_the_category(tmp_path: Path, category: str) -> None:
    inputs = tp.load_inputs(*_write(tmp_path, *_frames()))
    config = tp.make_config("lightgbm", inputs.spec, category, cutoffs=["2024-03-01"])
    dataset = tp.prepare(config, inputs, category)
    n_rows = int((inputs.train["Y_cat"] == category).sum())
    assert len(dataset.X) == n_rows
    assert not {"Y", "Y_raw", "Y_cat"} & set(dataset.X.columns)
    assert inputs.spec.columns == [c for c in dataset.X.columns if c.startswith("Y_")]
    train_idx, valid_idx = dataset.folds[0]
    assert train_idx.max() < valid_idx.min()
    assert dataset.test_frame is not None and "Y" not in dataset.test_frame.columns


def test_prepare_rejects_target_features_with_current_value(tmp_path: Path) -> None:
    train, test = _frames()
    # Y2 の行だけ、ラグのつもりで当日の値が入っている（リーク）CSV
    is_y2 = train["Y_cat"] == "Y2"
    train.loc[is_y2, "Y_lag_1"] = train.loc[is_y2, "Y"]
    inputs = tp.load_inputs(*_write(tmp_path, train, test))
    config = tp.make_config("lightgbm", inputs.spec, "Y2", cutoffs=["2024-03-01"])
    with pytest.raises(tp.LeakageError, match="一致しません"):
        tp.prepare(config, inputs, "Y2")
    # Y1 の行の特徴量は正しいので、Y1 は通る
    tp.prepare(tp.make_config("lightgbm", inputs.spec, "Y1", cutoffs=["2024-03-01"]), inputs, "Y1")


def test_make_config_rejects_unknown_model() -> None:
    spec = tp.parse_target_features(["Y_lag_1"])
    with pytest.raises(ValueError, match="モデル"):
        tp.make_config("random_forest", spec, "Y1")
    config = tp.make_config("xgboost", spec, "Y1")
    assert config.name == "x50_xgboost_Y1"
    assert config.data.target == "Y"
