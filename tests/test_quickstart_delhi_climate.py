"""scripts/quickstart_delhi_climate.py のテスト（生データが無い環境ではスキップ）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import quickstart_delhi_climate as qs
from pandas_helpers import as_list, row_at

pytestmark = pytest.mark.skipif(
    not (qs.TRAIN_PATH.is_file() and qs.TEST_PATH.is_file()),
    reason="data/raw の DailyDelhiClimate*.csv が無い（生データはgit管理外）",
)


def test_to_long_stacks_variables_as_series() -> None:
    wide = pd.DataFrame(
        {
            "date": [1, 2],
            "meantemp": [10.0, 11.0],
            "humidity": [80.0, 81.0],
            "meanpressure": [1015.0, 1016.0],
        }
    )
    long = qs.to_long(wide)
    assert list(long.columns) == ["date", "variable", "value"]
    assert len(long) == 2 * 3
    assert long.loc[long["variable"] == "humidity", "value"].pipe(as_list) == [80.0, 81.0]


def test_load_data_shapes_and_no_leak() -> None:
    train, test, truth = qs.load_data()
    assert list(train.columns) == ["date", "variable", "value"]
    # 予測対象には実測値を渡さない
    assert list(test.columns) == ["date", "variable"]
    assert set(train["variable"].unique()) == set(qs.TARGET_VARIABLES)  # wind_speed は含まない
    # 学習期間はテスト開始日より前で終わる（重なる行は除外）
    assert train["date"].max() < test["date"].min()
    assert len(test) == len(truth) == qs.HORIZON * len(qs.TARGET_VARIABLES)
    # 予測対象と実測値は同じ行順（予測値をそのまま実測値と比較できる）
    assert test.equals(truth[["date", "variable"]])


def test_training_pressure_outliers_are_interpolated() -> None:
    train, _, _ = qs.load_data()
    pressure = train.loc[train["variable"] == "meanpressure", "value"]
    low, high = qs.PRESSURE_VALID_RANGE
    assert pressure.isna().sum() == 0
    assert pressure.between(low, high).all()


def test_interpolate_invalid_pressure() -> None:
    wide = pd.DataFrame({"meanpressure": [1000.0, 7679.0, 1010.0, -3.0]})
    out = qs.interpolate_invalid_pressure(wide)["meanpressure"].pipe(as_list)
    # 途中の異常値は前後の直線補間、末尾は直前の値で埋める
    assert out == pytest.approx([1000.0, 1005.0, 1010.0, 1010.0])


def test_evaluation_excludes_invalid_pressure() -> None:
    truth = pd.DataFrame(
        {
            "date": [1, 1, 2],
            "variable": ["meantemp", "meanpressure", "meanpressure"],
            "value": [10.0, 59.0, 1000.0],
        }
    )
    scores = qs.evaluate_on_test(truth, {"m": np.array([11.0, 1000.0, 1002.0])})
    pressure = row_at(scores[scores["variable"] == "meanpressure"], 0)
    # 59 hPa の行は除外され、残り1行のMAE=2になる
    assert pressure["n"] == 1
    assert pressure["test_mae"] == pytest.approx(2.0)


def test_experiment_config_is_valid() -> None:
    config = qs.make_experiment_config("lightgbm", n_trials=3)
    assert config.forecast is not None
    assert config.forecast.series_col == "variable"
    assert config.forecast.horizon == qs.HORIZON
    # 365日の季節差分（前年同日との差）を予測する
    assert config.forecast.target_transform == "seasonal_diff"
    assert config.forecast.seasonal_period == 365
    assert config.tuning.enabled and config.tuning.n_trials == 3
    assert config.metrics == ["mae"]
    # 誤差評価: 3系列すべてを描き、学習曲線・検証曲線も作る
    assert config.evaluation.enabled
    assert config.evaluation.max_series == len(qs.TARGET_VARIABLES)
    assert config.evaluation.learning_curve.enabled
    assert config.evaluation.validation_curve.param == "learning_rate"
    assert config.evaluation.ljung_box_lags == qs.LJUNG_BOX_LAGS


def test_evaluate_on_test_per_variable_and_overall() -> None:
    truth = pd.DataFrame(
        {
            "date": [1, 1, 1],
            "variable": ["meantemp", "humidity", "meanpressure"],
            "value": [10.0, 50.0, 1000.0],
        }
    )
    scores = qs.evaluate_on_test(truth, {"m": np.array([11.0, 52.0, 1003.0])})
    # テスト用に日付以外を与える場合も、気圧1000は有効範囲内なので全行が評価に使われる
    by_variable = dict(zip(scores["variable"], scores["test_mae"], strict=True))
    assert by_variable == pytest.approx(
        {"meantemp": 1.0, "humidity": 2.0, "meanpressure": 3.0, "all": 2.0}
    )


def test_quickstart_end_to_end(tmp_path: Path) -> None:
    qs.main(["--n-trials", "1", "--no-tracking", "--output-root", str(tmp_path)])
    scores = pd.read_csv(tmp_path / "tables" / "delhi_quickstart__test_mae.csv")
    assert set(scores["model"]) == {"lightgbm", "xgboost", "ensemble"}
    assert set(scores["variable"]) == {*qs.TARGET_VARIABLES, "all"}
    assert scores["test_mae"].is_finite().all()
    assert (tmp_path / "figures" / "delhi_quickstart__test_forecast.png").is_file()
    for model in qs.MODELS:
        run_dir = next((tmp_path / "experiments" / f"delhi_{model}").iterdir())
        for name in ("horizon_scores.csv", "tuning_trials.csv", "test_predictions.parquet"):
            assert (run_dir / name).is_file(), f"{model}: {name}"
        assert (run_dir / "shap" / "shap_beeswarm.png").is_file()
        # 誤差評価（残差の診断・学習の推移・学習曲線・検証曲線）
        evaluation = {p.name for p in (run_dir / "evaluation").iterdir()}
        assert {
            "residual_summary.csv",
            "residual_distribution.png",
            "qq_plot.png",
            "residual_acf_pacf.png",
            "residual_plot.png",
            "leverage_cooks_distance.png",
            "training_history.png",
            "learning_curve.png",
            "validation_curve.png",
        } <= evaluation
        # 時系列の残差診断（再帰予測と1日先予測の残差、系列ごと）
        assert {
            "residual_tests.csv",
            "ljung_box.csv",
            "residual_timeseries.png",
            "residual_timeseries_onestep.png",
        } <= evaluation
        for kind in ("recursive", "onestep"):
            for variable in qs.TARGET_VARIABLES:
                assert f"residual_diagnostics__{kind}__{variable}.png" in evaluation
        tests = pd.read_csv(run_dir / "evaluation" / "residual_tests.csv")
        assert len(tests) == 2 * len(qs.TARGET_VARIABLES)
        summary = pd.read_csv(run_dir / "evaluation" / "residual_summary.csv")
        assert summary["series"].pipe(as_list) == ["all", *sorted(qs.TARGET_VARIABLES)]
        # テスト予測は 114日 × 3変数
        test_pred = pd.read_parquet(run_dir / "test_predictions.parquet")
        assert len(test_pred) == qs.HORIZON * len(qs.TARGET_VARIABLES)
    assert any((tmp_path / "ensembles" / "delhi_blend").iterdir())
