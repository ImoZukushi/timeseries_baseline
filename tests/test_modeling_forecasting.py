"""modeling.forecasting（再帰的多段予測）のテスト。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas_helpers import as_list, row_at
from pydantic import ValidationError

from modeling.config import ExperimentConfig, ForecastConfig
from modeling.experiment import prepare_dataset, run_experiment
from modeling.forecasting import (
    TargetFeatureBuilder,
    make_builder,
    recursive_backtest,
    recursive_forecast,
    step_index,
)
from modeling.trainer import fit_pipeline

START = dt.date(2024, 1, 1)


def _panel(n_days: int = 120, seed: int = 0, noise: float = 1.0) -> pd.DataFrame:
    """2系列のパネルデータ（AR(1) + 週次の季節性）。曜日は将来も既知の外生変数。"""
    rng = np.random.default_rng(seed)
    rows = []
    for sid, level in (("A", 20.0), ("B", 50.0)):
        y = level
        for t in range(n_days):
            y = 0.7 * y + 0.3 * level + 5 * np.sin(2 * np.pi * t / 7) + rng.normal() * noise
            rows.append(
                {"date": START + dt.timedelta(days=t), "sid": sid, "dow": t % 7, "y": float(y)}
            )
    return (
        pd.DataFrame(rows)
        .assign(date=lambda d: pd.to_datetime(d["date"]))
        .sort_values(["date", "sid"], kind="stable")
        .reset_index(drop=True)
    )


def _raw_config(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "name": "fc",
        "task": "time_series",
        "data": {
            "train_path": "unused.csv",
            "target": "y",
            "time_col": "date",
            "drop_cols": ["date"],
        },
        "features": [
            {
                "class": "feature_engineering.categorical.PandasOrdinalEncoder",
                "params": {"variables": ["sid"]},
            }
        ],
        "cv": {"method": "time_cutoff", "cutoffs": ["2024-03-01", "2024-04-01"]},
        "model": {"name": "lightgbm", "params": {"n_estimators": 50}},
        "metrics": ["rmse", "mae"],
        "forecast": {"series_col": "sid", "lags": [1, 2, 7], "rolling_windows": [7]},
        "explain": {"enabled": False},
    }
    return raw | overrides


def _config(**overrides: Any) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_raw_config(**overrides))


# --- 設定 ------------------------------------------------------------------------------


def test_forecast_config_validation() -> None:
    with pytest.raises(ValidationError, match="time_series"):
        _config(task="regression", cv={"method": "kfold"})
    raw = _raw_config()
    raw["data"].pop("time_col")
    raw["cv"] = {"method": "time_series"}
    with pytest.raises(ValidationError, match="time_col"):
        ExperimentConfig.model_validate(raw)
    with pytest.raises(ValidationError, match="1以上"):
        ForecastConfig(lags=[0, 1])
    with pytest.raises(ValidationError, match="lags に 1"):
        ForecastConfig(lags=[2], rolling_windows=[3])
    with pytest.raises(ValidationError):
        ForecastConfig.model_validate({"lags": [1], "clip": {"min": 1, "max": 0}})


# --- 目的変数の特徴量 -----------------------------------------------------------------


def _builder(**kwargs: Any) -> TargetFeatureBuilder:
    return TargetFeatureBuilder("y", "t", ForecastConfig(**kwargs))


def test_target_features_use_only_past_values() -> None:
    frame = pd.DataFrame({"t": [1, 2, 3, 4, 5], "y": [1.0, 2.0, 3.0, 4.0, 5.0]})
    builder = _builder(lags=[1, 2], rolling_windows=[2], rate_of_change=True)
    out = builder.build(frame)
    assert builder.feature_names == ["y_lag_1", "y_lag_2", "y_lag_1_ma_2", "y_lag_1_roc_1"]
    row = row_at(out, 3)  # y=4 の行
    assert row["y_lag_1"] == 3.0
    assert row["y_lag_2"] == 2.0
    assert row["y_lag_1_ma_2"] == pytest.approx(2.5)  # (y_{t-1} + y_{t-2}) / 2
    assert row["y_lag_1_roc_1"] == pytest.approx(0.5)  # (3 - 2) / 2
    # 行tの値を変えても行tの特徴量は変わらない（y_t を含まない）
    changed = builder.build(frame.assign(y=[1.0, 2.0, 3.0, 999.0, 5.0]))
    assert row_at(changed, 3)["y_lag_1_ma_2"] == pytest.approx(2.5)


def test_target_features_per_series_and_order_preserved() -> None:
    frame = pd.DataFrame(
        {"s": ["A", "B", "A", "B"], "t": [2, 2, 1, 1], "y": [2.0, 20.0, 1.0, 10.0]}
    )
    builder = TargetFeatureBuilder("y", "t", ForecastConfig(series_col="s", lags=[1]))
    out = builder.build(frame)
    # 入力の行順のまま、系列内の時刻順でラグが付く
    assert out["y_lag_1"].pipe(as_list) == [1.0, 10.0, None, None]
    assert builder.lookback == 1


def test_target_features_with_missing_values() -> None:
    frame = pd.DataFrame({"t": [1, 2, 3, 4, 5], "y": [1.0, np.nan, 3.0, 4.0, 5.0]})
    # 既定（min_periods = 窓幅）では、窓内に欠損があると移動平均も欠損
    out = _builder(lags=[1], rolling_windows=[3]).build(frame)
    assert row_at(out, 3)["y_lag_1_ma_3"] is None
    # rolling_min_periods=1 なら、窓内の観測値だけで平均する（y_{t-1}, y_{t-3} の平均）
    out = _builder(lags=[1], rolling_windows=[3], rolling_min_periods=1).build(frame)
    assert row_at(out, 3)["y_lag_1_ma_3"] == pytest.approx(2.0)
    # ラグは欠損のまま（埋めない）
    assert row_at(out, 2)["y_lag_1"] is None
    # 窓幅より大きい min_periods は窓幅として扱う
    out = _builder(lags=[1], rolling_windows=[2], rolling_min_periods=5).build(frame)
    assert row_at(out, 4)["y_lag_1_ma_2"] == pytest.approx(3.5)


def test_lookback() -> None:
    assert _builder(lags=[1, 3], rolling_windows=[5]).lookback == 5
    assert _builder(lags=[1], rate_of_change=True).lookback == 2


def test_step_index() -> None:
    frame = pd.DataFrame({"s": ["A", "B", "A", "A", "B"], "t": [5, 9, 3, 4, 8]})
    assert step_index(frame, "t", "s").tolist() == [3, 2, 1, 2, 1]
    assert step_index(frame, "t", None).tolist() == [3, 5, 1, 2, 4]


# --- 再帰予測 --------------------------------------------------------------------------


def _ar_frame(n: int = 60) -> pd.DataFrame:
    """y_t = 0.9 * y_{t-1} の決定的な系列。"""
    return pd.DataFrame({"t": np.arange(n), "y": 100.0 * 0.9 ** np.arange(n)})


def _linear_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "ar",
            "task": "time_series",
            "data": {"train_path": "unused.csv", "target": "y", "time_col": "t"},
            "cv": {"method": "time_series", "n_splits": 2},
            "model": {"name": "linear", "params": {"alpha": 1e-10}},
            "metrics": ["rmse"],
            "forecast": {"lags": [1]},
        }
    )


def _fit_ar(frame: pd.DataFrame, config: ExperimentConfig) -> Any:
    builder = make_builder(config)
    X = builder.build(frame)[builder.feature_names]
    # 先頭行はラグが無いので学習から除く（決定的な関係を正確に学習させるため）
    return fit_pipeline(config, X[1:], frame["y"].to_numpy()[1:]), builder, X.columns


def test_recursive_forecast_reproduces_deterministic_trajectory() -> None:
    config = _linear_config()
    frame = _ar_frame()
    pipeline, builder, columns = _fit_ar(frame[:40], config)
    pred = recursive_forecast(pipeline, frame[:40], frame[40:55], builder, columns)
    # 予測値を次のラグとして使い続けるので、15期先まで真の軌跡をたどる
    assert pred == pytest.approx(frame["y"].to_numpy()[40:55], rel=1e-6)


def test_recursive_forecast_ignores_future_target_values() -> None:
    config = _linear_config()
    frame = _ar_frame()
    pipeline, builder, columns = _fit_ar(frame[:40], config)
    future = frame[40:50]
    garbage = future.assign(y=-1e9)
    no_target = future.drop(columns="y")
    base = recursive_forecast(pipeline, frame[:40], future, builder, columns)
    assert recursive_forecast(pipeline, frame[:40], garbage, builder, columns) == pytest.approx(
        base
    )
    assert recursive_forecast(pipeline, frame[:40], no_target, builder, columns) == pytest.approx(
        base
    )


def test_recursive_forecast_clip_is_applied_during_recursion() -> None:
    raw = _linear_config().model_dump(mode="json", by_alias=True)
    raw["forecast"]["clip"] = {"min": 0.0}
    config = ExperimentConfig.model_validate(raw)
    # 毎期10ずつ減る系列 → 線形外挿は負になるが、clipで0に止まる
    frame = pd.DataFrame({"t": np.arange(30), "y": 200.0 - 10.0 * np.arange(30)})
    builder = make_builder(config)
    X = builder.build(frame)[builder.feature_names]
    pipeline = fit_pipeline(config, X[1:], frame["y"].to_numpy()[1:])
    future = pd.DataFrame({"t": np.arange(30, 60)})
    pred = recursive_forecast(pipeline, frame, future, builder, X.columns)
    assert (pred >= 0).all()
    assert pred[-1] == pytest.approx(0.0, abs=1e-6)


def test_recursive_forecast_series_are_independent() -> None:
    config = _config()
    ds = prepare_dataset(config, _panel())
    assert ds.frame is not None
    pipeline = fit_pipeline(config, ds.X, ds.y)
    builder = make_builder(config)
    cutoff = pd.Timestamp(2024, 4, 1)
    history = ds.frame[ds.frame["date"] < cutoff]
    future = ds.frame[ds.frame["date"] >= cutoff]
    base = recursive_forecast(pipeline, history, future, builder, ds.X.columns)
    # 系列Bの履歴を書き換えても、系列Aの予測は変わらない
    shifted = history.assign(y=history["y"].where(history["sid"] != "B", history["y"] + 1000))
    changed = recursive_forecast(pipeline, shifted, future, builder, ds.X.columns)
    is_a = (future["sid"] == "A").to_numpy()
    assert changed[is_a] == pytest.approx(base[is_a])
    assert not np.allclose(changed[~is_a], base[~is_a])


def test_recursive_forecast_requires_future_after_history() -> None:
    config = _linear_config()
    frame = _ar_frame()
    pipeline, builder, columns = _fit_ar(frame[:40], config)
    with pytest.raises(ValueError, match="履歴"):
        recursive_forecast(pipeline, frame[:40], frame[30:45], builder, columns)


# --- 再帰バックテスト ----------------------------------------------------------------


def test_backtest_does_not_use_validation_targets() -> None:
    config = _config(cv={"method": "time_cutoff", "cutoffs": ["2024-04-01"]})
    frame = _panel()
    cutoff = pd.Timestamp(2024, 4, 1)
    ds = prepare_dataset(config, frame)
    assert ds.frame is not None
    base = recursive_backtest(config, ds.frame, ds.X, ds.y, ds.folds)
    # 検証期間の目的変数を乱数に置き換えても、再帰予測（OOF）は変わらない
    rng = np.random.default_rng(1)
    noise = pd.Series(rng.normal(size=len(frame)) * 1000, index=frame.index)
    tampered = frame.assign(y=frame["y"].where(frame["date"] < cutoff, noise))
    ds2 = prepare_dataset(config, tampered)
    assert ds2.frame is not None
    again = recursive_backtest(config, ds2.frame, ds2.X, ds2.y, ds2.folds)
    valid = base.fold_ids >= 0
    assert again.oof_pred[valid] == pytest.approx(base.oof_pred[valid])
    # 1期先予測は検証期間の真のラグを使うので変わる（＝再帰予測との違い）
    assert not np.allclose(again.onestep_oof_pred[valid], base.onestep_oof_pred[valid])


def test_backtest_horizon_truncation_and_scores() -> None:
    raw = _raw_config()
    raw["forecast"]["horizon"] = 5
    config = ExperimentConfig.model_validate(raw)
    ds = prepare_dataset(config, _panel())
    assert ds.frame is not None
    result = recursive_backtest(config, ds.frame, ds.X, ds.y, ds.folds)
    predicted = result.fold_ids >= 0
    # 2fold × 2系列 × 5ステップ
    assert predicted.sum() == 2 * 2 * 5
    assert np.isnan(result.oof_pred[~predicted]).all()
    assert set(result.steps[predicted].tolist()) == {1, 2, 3, 4, 5}
    hs = result.horizon_scores
    assert hs["step"].pipe(as_list) == [1, 2, 3, 4, 5]
    assert hs["n"].pipe(as_list) == [4] * 5
    assert set(hs.columns) == {"step", "n", "rmse", "mae"}


def test_backtest_skips_missing_targets() -> None:
    raw = _raw_config()
    raw["forecast"]["rolling_min_periods"] = 1
    config = ExperimentConfig.model_validate(raw)
    frame = _panel()
    # 散発的な欠損と、検証期間にかかる連続した欠損を入れる
    missing = np.zeros(len(frame), dtype=bool)
    missing[::17] = True
    missing[(frame["date"] >= "2024-03-10") & (frame["date"] < "2024-03-20")] = True
    frame = frame.assign(y=frame["y"].mask(missing))
    ds = prepare_dataset(config, frame)
    assert ds.frame is not None and np.isnan(ds.y).sum() == missing.sum()
    result = recursive_backtest(config, ds.frame, ds.X, ds.y, ds.folds)
    predicted = result.fold_ids >= 0
    # 欠損の日も再帰予測はする（予測値を次のラグに使う）。評価は観測値の行だけで行う
    assert np.isfinite(result.oof_pred[predicted]).all()
    assert all(np.isfinite(s) for s in result.fold_scores["mae"])
    observed = predicted & ~missing
    mae = np.mean(np.abs(ds.y[observed] - result.oof_pred[observed]))
    assert result.oof_scores["mae"] == pytest.approx(mae)
    assert result.horizon_scores["n"].sum() == observed.sum()
    # 再帰予測でない実験では、目的変数の欠損はエラーのまま
    with pytest.raises(ValueError, match="欠損"):
        prepare_dataset(
            _config(task="regression", forecast=None, cv={"method": "kfold", "n_splits": 2}),
            frame,
        )


def test_backtest_recursive_error_exceeds_one_step_error() -> None:
    config = _config()
    ds = prepare_dataset(config, _panel(noise=3.0))
    assert ds.frame is not None
    result = recursive_backtest(config, ds.frame, ds.X, ds.y, ds.folds)
    # 誤差が蓄積するので、真のラグを使う1期先予測より再帰予測の誤差の方が大きい
    assert result.oof_scores["rmse"] > result.onestep_oof_scores["rmse"]
    hs = result.horizon_scores
    assert hs.loc[hs["step"] <= 3, "rmse"].mean() < hs.loc[hs["step"] >= 20, "rmse"].mean()


# --- 実験・CLI --------------------------------------------------------------------------


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    full = _panel(n_days=130)
    cutoff = pd.Timestamp(2024, 4, 30)
    train = full[full["date"] < cutoff]
    test = full[full["date"] >= cutoff].drop(columns="y")
    train_path, test_path = tmp_path / "train.parquet", tmp_path / "test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    return train_path, test_path


@pytest.mark.parametrize("test_prediction", ["fold_mean", "refit_full"])
def test_run_experiment_forecast_outputs(tmp_path: Path, test_prediction: str) -> None:
    import mlflow

    from modeling.tracking import MLflowTracker

    train_path, test_path = _write_inputs(tmp_path)
    raw = _raw_config(test_prediction=test_prediction, explain={"enabled": True, "max_samples": 50})
    raw["data"].update(train_path=str(train_path), test_path=str(test_path))
    config = ExperimentConfig.model_validate(raw)
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    tracker = MLflowTracker("fc", tracking_uri=uri, artifact_root=tmp_path / "art")
    result = run_experiment(config, tracker=tracker, output_root=tmp_path / "out")
    out = result.output_dir
    for name in ("horizon_scores.csv", "horizon_error.png", "test_predictions.parquet"):
        assert (out / name).is_file(), name
    assert (out / "shap" / "shap_beeswarm.png").is_file()
    test_pred = pd.read_parquet(out / "test_predictions.parquet")
    assert len(test_pred) == 2 * 10  # 2系列 × 10日
    assert test_pred["pred"].isna().sum() == 0
    logged = mlflow.get_run(result.run_id).data.metrics
    assert "oof_rmse" in logged and "onestep_oof_rmse" in logged


def test_run_experiment_script_forecast_with_tune(tmp_path: Path) -> None:
    import run_experiment as script

    train_path, test_path = _write_inputs(tmp_path)
    raw = _raw_config()
    raw["data"].update(train_path=str(train_path), test_path=str(test_path))
    config_path = tmp_path / "fc.yaml"
    config_path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    script.main(
        [
            "--config",
            str(config_path),
            "--no-tracking",
            "--tune",
            "--n-trials",
            "2",
            "--optuna-dir",
            str(tmp_path / "optuna"),
            "--output-root",
            str(tmp_path / "out"),
        ]
    )
    assert next((tmp_path / "out").rglob("horizon_scores.csv")).is_file()
    assert next((tmp_path / "out").rglob("tuning_trials.csv")).is_file()
