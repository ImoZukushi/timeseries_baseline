"""scripts/quickstart_delhi_climate.py のテスト（生データが無い環境ではスキップ）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import quickstart_delhi_climate as qs

pytestmark = pytest.mark.skipif(
    not (qs.TRAIN_PATH.is_file() and qs.TEST_PATH.is_file()),
    reason="data/raw の DailyDelhiClimate*.csv が無い（生データはgit管理外）",
)


def test_to_long_stacks_variables_as_series() -> None:
    wide = pl.DataFrame(
        {
            "date": [1, 2],
            "meantemp": [10.0, 11.0],
            "humidity": [80.0, 81.0],
            "meanpressure": [1015.0, 1016.0],
        }
    )
    long = qs.to_long(wide)
    assert long.columns == ["date", "variable", "value"]
    assert long.height == 2 * 3
    assert long.filter(pl.col("variable") == "humidity")["value"].to_list() == [80.0, 81.0]


def test_load_data_shapes_and_no_leak() -> None:
    train, test, truth = qs.load_data()
    assert train.columns == ["date", "variable", "value"]
    # 予測対象には実測値を渡さない
    assert test.columns == ["date", "variable"]
    assert set(train["variable"].unique()) == set(qs.TARGET_VARIABLES)  # wind_speed は含まない
    # 学習期間はテスト開始日より前で終わる（重なる行は除外）
    assert train["date"].max() < test["date"].min()
    assert test.height == truth.height == qs.HORIZON * len(qs.TARGET_VARIABLES)
    # 予測対象と実測値は同じ行順（予測値をそのまま実測値と比較できる）
    assert test.equals(truth.select("date", "variable"))


def test_training_pressure_outliers_are_interpolated() -> None:
    train, _, _ = qs.load_data()
    pressure = train.filter(pl.col("variable") == "meanpressure")["value"]
    low, high = qs.PRESSURE_VALID_RANGE
    assert pressure.null_count() == 0
    assert pressure.is_between(low, high).all()


def test_interpolate_invalid_pressure() -> None:
    wide = pl.DataFrame({"meanpressure": [1000.0, 7679.0, 1010.0, -3.0]})
    out = qs.interpolate_invalid_pressure(wide)["meanpressure"].to_list()
    # 途中の異常値は前後の直線補間、末尾は直前の値で埋める
    assert out == pytest.approx([1000.0, 1005.0, 1010.0, 1010.0])


def test_evaluation_excludes_invalid_pressure() -> None:
    truth = pl.DataFrame(
        {
            "date": [1, 1, 2],
            "variable": ["meantemp", "meanpressure", "meanpressure"],
            "value": [10.0, 59.0, 1000.0],
        }
    )
    scores = qs.evaluate_on_test(truth, {"m": np.array([11.0, 1000.0, 1002.0])})
    pressure = scores.filter(pl.col("variable") == "meanpressure").row(0, named=True)
    # 59 hPa の行は除外され、残り1行のMAE=2になる
    assert pressure["n"] == 1
    assert pressure["test_mae"] == pytest.approx(2.0)


def test_experiment_config_is_valid() -> None:
    config = qs.make_experiment_config("lightgbm", n_trials=3)
    assert config.forecast is not None
    assert config.forecast.series_col == "variable"
    assert config.forecast.horizon == qs.HORIZON
    assert config.tuning.enabled and config.tuning.n_trials == 3
    assert config.metrics == ["mae"]


def test_evaluate_on_test_per_variable_and_overall() -> None:
    truth = pl.DataFrame(
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
    scores = pl.read_csv(tmp_path / "tables" / "delhi_quickstart__test_mae.csv")
    assert set(scores["model"]) == {"lightgbm", "xgboost", "ensemble"}
    assert set(scores["variable"]) == {*qs.TARGET_VARIABLES, "all"}
    assert scores["test_mae"].is_finite().all()
    assert (tmp_path / "figures" / "delhi_quickstart__test_forecast.png").is_file()
    for model in qs.MODELS:
        run_dir = next((tmp_path / "experiments" / f"delhi_{model}").iterdir())
        for name in ("horizon_scores.csv", "tuning_trials.csv", "test_predictions.parquet"):
            assert (run_dir / name).is_file(), f"{model}: {name}"
        assert (run_dir / "shap" / "shap_beeswarm.png").is_file()
        # テスト予測は 114日 × 3変数
        test_pred = pl.read_parquet(run_dir / "test_predictions.parquet")
        assert test_pred.height == qs.HORIZON * len(qs.TARGET_VARIABLES)
    assert any((tmp_path / "ensembles" / "delhi_blend").iterdir())
