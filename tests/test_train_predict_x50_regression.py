"""scripts/train_predict_x50_regression.py（通常の回帰・GroupKFold）の入力とリーク防止のテスト。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import train_predict_x50 as base
import train_predict_x50_regression as reg
from test_train_predict_x50 import CATEGORIES, _frames, _write


def _inputs(tmp_path: Path, missing: bool = False) -> base.Inputs:
    return base.load_inputs(*_write(tmp_path, *_frames(missing=missing)))


def test_model_frames_drop_target_features_and_missing_targets(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, missing=True)
    train, test = reg.model_frames(inputs)
    # 目的変数は元の尺度（Y_raw）。欠損の行は除く。目的変数の特徴量（ラグ等）は使わない
    observed = inputs.train["Y_raw"].notna()
    assert len(train) == int(observed.sum())
    assert train["Y"].tolist() == pytest.approx(inputs.train.loc[observed, "Y_raw"].tolist())
    assert not [c for c in train.columns if c.startswith("Y_lag")]
    # グループは年月
    assert set(train["year_month"]) == {"2024-01", "2024-02", "2024-03"}
    assert "Y" not in test.columns and len(test) == len(inputs.test)


def test_prepare_uses_group_kfold_by_month(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    config = reg.make_config("lightgbm", len(CATEGORIES), n_splits=3)
    dataset, frame = reg.prepare(config, inputs)
    assert len(dataset.X) == len(frame)
    # 特徴量は系列ID・日付の特徴量・説明変数だけ（目的変数・日付・グループ・ラグは無い）
    assert set(dataset.X.columns) == {"Y_cat", *inputs.exog_cols}
    # 同じ月が学習と検証に分かれず、各月はちょうど1つの fold の検証に入る
    months = frame["year_month"].to_numpy()
    validated = []
    for train_idx, valid_idx in dataset.folds:
        assert not set(months[train_idx]) & set(months[valid_idx])
        validated += sorted(set(months[valid_idx]))
    assert sorted(validated) == sorted(set(months))
    assert dataset.X_test is not None and "Y" not in dataset.X_test.columns


def test_make_config_is_plain_regression() -> None:
    config = reg.make_config("xgboost", 3)
    assert config.task == "regression" and config.forecast is None
    assert config.cv.method == "group" and config.data.group_col == "year_month"
    assert config.data.target == "Y"
    with pytest.raises(ValueError, match="モデル"):
        reg.make_config("random_forest", 3)


def test_collect_predictions_skips_missing_answers(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    config = reg.make_config("lightgbm", len(CATEGORIES), n_splits=3)
    dataset, frame = reg.prepare(config, inputs)

    class _Result:
        """`collect_predictions` が使う属性だけを持つ、run_experiment の結果の代わり。"""

        class cv_result:  # noqa: N801
            test_pred = np.full(len(inputs.test), 50.0)
            oof_pred = np.full(len(frame), 1.0)
            fold_ids = np.zeros(len(frame), dtype=int)

    answer = inputs.test[["date", "Y_cat"]].assign(Y=52.0)
    answer.loc[answer.index[0], "Y"] = np.nan
    run = reg.collect_predictions(
        inputs.categories,
        dataset,
        frame,
        inputs.test,
        _Result(),
        answer,
    )
    assert run.predictions["actual"].isna().sum() == 1
    assert list(run.oof.columns) == ["date", "Y_cat", "actual", "pred", "fold"]
    # 正解の日付・種類が足りない場合はエラー
    with pytest.raises(ValueError, match="正解ファイル"):
        reg.collect_predictions(
            inputs.categories,
            dataset,
            frame,
            inputs.test,
            _Result(),
            answer.iloc[1:],
        )
    summary = pd.DataFrame(reg.summary_frame(run))
    # 正解が欠損の行は除いて MAE を計算する（予測 50・正解 52 → 2）
    assert summary["test_mae"].tolist() == pytest.approx([2.0, 2.0])
