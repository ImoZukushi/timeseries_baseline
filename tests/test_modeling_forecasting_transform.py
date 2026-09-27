"""再帰予測の目的変数の変換（forecast.target_transform）のテスト。"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from modeling.config import EnsembleConfig, ExperimentConfig, ForecastConfig
from modeling.dataset import cross_validate
from modeling.ensemble import run_ensemble
from modeling.experiment import prepare_dataset, run_experiment

START = dt.date(2024, 1, 1)
CUTOFF = "2024-06-01"  # 検証期間の開始日（学習は152日分）


def _dates(n: int) -> list[dt.date]:
    return [START + dt.timedelta(days=i) for i in range(n)]


def _config(transform: str | None, model: str = "lightgbm", **forecast: Any) -> ExperimentConfig:
    raw: dict[str, Any] = {
        "name": f"tt_{transform}",
        "task": "time_series",
        "data": {
            "train_path": "unused.csv",
            "target": "y",
            "time_col": "date",
            "drop_cols": ["date"],
        },
        "cv": {"method": "time_cutoff", "cutoffs": [CUTOFF]},
        "model": {"name": model, "params": {"n_estimators": 100}},
        "metrics": ["mae"],
        "forecast": {"lags": [1, 2], "target_transform": transform, **forecast},
        "explain": {"enabled": False},
    }
    return ExperimentConfig.model_validate(raw)


def _backtest(config: ExperimentConfig, frame: pl.DataFrame) -> Any:
    return cross_validate(config, prepare_dataset(config, frame))


# --- 設定 ------------------------------------------------------------------------------


def test_seasonal_transform_requires_period() -> None:
    with pytest.raises(ValidationError, match="seasonal_period"):
        ForecastConfig(lags=[1], target_transform="seasonal_diff")
    with pytest.raises(ValidationError):
        ForecastConfig(lags=[1], target_transform="box_cox")  # type: ignore[arg-type]
    assert ForecastConfig(lags=[1], target_transform="seasonal_diff", seasonal_period=7)


# --- 変換の効果 --------------------------------------------------------------------------


def test_diff_lets_gbdt_extrapolate_a_trend() -> None:
    # 直線的に増え続ける系列。GBDTは学習範囲の外（より大きい値）を予測できないが、
    # 差分（毎日 +0.5 で一定）を予測して元に戻せば、そのまま外挿できる
    n = 200
    frame = pl.DataFrame({"date": _dates(n), "y": 10 + 0.5 * np.arange(n)})
    raw = _backtest(_config(None), frame)
    diff = _backtest(_config("diff"), frame)
    assert raw.oof_scores["mae"] > 5.0
    assert diff.oof_scores["mae"] < 0.01
    # OOFは元の尺度で保存される（検証期間の真の値に近い）
    valid = diff.fold_ids >= 0
    assert diff.oof_pred[valid] == pytest.approx(frame["y"].to_numpy()[valid], abs=0.01)


def test_log_seasonal_diff_restores_multiplicative_seasonality() -> None:
    # 指数的な成長 × 週周期の季節性（乗法的）。対数季節差分は一定値になる
    n = 210
    t = np.arange(n)
    y = 100 * 1.01**t * (1 + 0.3 * np.sin(2 * np.pi * t / 7))
    frame = pl.DataFrame({"date": _dates(n), "y": y})
    result = _backtest(_config("log_seasonal_diff", seasonal_period=7), frame)
    valid = result.fold_ids >= 0
    assert result.oof_pred[valid] == pytest.approx(y[valid], rel=1e-3)


def test_one_step_scores_are_on_original_scale() -> None:
    n = 200
    frame = pl.DataFrame({"date": _dates(n), "y": 10 + 0.5 * np.arange(n)})
    result = _backtest(_config("diff"), frame)
    assert np.isfinite(result.onestep_oof_scores["mae"])
    assert result.onestep_oof_scores["mae"] < 0.01


# --- リーク・整合性 ----------------------------------------------------------------------


def _panel(n: int = 200, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sid, level, slope in (("A", 20.0, 0.1), ("B", 50.0, -0.05)):
        for i, day in enumerate(_dates(n)):
            y = level + slope * i + 3 * np.sin(2 * np.pi * i / 7) + rng.normal()
            rows.append({"date": day, "sid": sid, "y": float(y)})
    return pl.DataFrame(rows).sort("date", "sid")


def _panel_config(transform: str | None, **forecast: Any) -> ExperimentConfig:
    raw = _config(transform, series_col="sid", **forecast).model_dump(mode="json", by_alias=True)
    raw["features"] = [
        {
            "class": "feature_engineering.categorical.PolarsOrdinalEncoder",
            "params": {"variables": ["sid"]},
        }
    ]
    return ExperimentConfig.model_validate(raw)


@pytest.mark.parametrize("transform", ["diff", "log_seasonal_diff"])
def test_backtest_with_transform_does_not_use_validation_targets(transform: str) -> None:
    config = _panel_config(transform, seasonal_period=7)
    frame = _panel()
    base = _backtest(config, frame)
    rng = np.random.default_rng(1)
    tampered = frame.with_columns(
        pl.when(pl.col("date") >= dt.date(2024, 6, 1))
        .then(pl.lit(rng.uniform(10, 1000, frame.height)))
        .otherwise(pl.col("y"))
        .alias("y")
    )
    again = _backtest(config, tampered)
    valid = base.fold_ids >= 0
    assert again.oof_pred[valid] == pytest.approx(base.oof_pred[valid])


def test_different_transforms_share_rows_and_can_be_ensembled(tmp_path: Path) -> None:
    # 学習は200日、テストはその後の5日（2系列）
    full = _panel(n=205)
    frame = full.filter(pl.col("date") < START + dt.timedelta(days=200))
    test = full.filter(pl.col("date") >= START + dt.timedelta(days=200)).drop("y")
    results = {}
    for name, transform in (("diff", "diff"), ("seasonal", "seasonal_diff")):
        config = _panel_config(transform, seasonal_period=7)
        results[name] = run_experiment(
            config, prepare_dataset(config, frame, test), output_root=tmp_path
        )
    # 差分で計算できない先頭行の数は違っても、OOFの行・fold番号はそろう
    ids = [r.cv_result.fold_ids for r in results.values()]
    assert np.array_equal(ids[0], ids[1])
    ensemble = run_ensemble(
        EnsembleConfig.model_validate(
            {
                "name": "tt_blend",
                "task": "time_series",
                "members": [{"name": k, "path": str(r.output_dir)} for k, r in results.items()],
                "method": "mean",
                "metrics": ["mae"],
            }
        ),
        output_root=tmp_path / "ens",
    )
    assert np.isfinite(ensemble.scores["mae"].to_numpy()).all()


# --- 範囲制限・対数・テスト予測 ------------------------------------------------------------


def test_clip_applies_on_original_scale() -> None:
    # 毎日2ずつ減る系列。差分をそのまま外挿すると負になるが、元の尺度で0に止まる
    n = 200
    frame = pl.DataFrame({"date": _dates(n), "y": 350.0 - 2.0 * np.arange(n)})
    clipped = _backtest(_config("diff", clip={"min": 0.0}), frame)
    valid = clipped.fold_ids >= 0
    assert (clipped.oof_pred[valid] >= 0).all()
    assert clipped.oof_pred[valid][-1] == pytest.approx(0.0)


def test_log_transform_requires_positive_target_unless_offset() -> None:
    n = 200
    frame = pl.DataFrame({"date": _dates(n), "y": np.r_[0.0, np.arange(1, n, dtype=float)]})
    with pytest.raises(ValueError, match="target_offset"):
        prepare_dataset(_config("log_diff"), frame)
    result = _backtest(_config("log_diff", target_offset=1.0), frame)
    assert np.isfinite(result.oof_scores["mae"])


def test_run_experiment_test_predictions_on_original_scale(tmp_path: Path) -> None:
    n, horizon = 200, 20
    full = pl.DataFrame({"date": _dates(n + horizon), "y": 10 + 0.5 * np.arange(n + horizon)})
    train, test = full.head(n), full.tail(horizon)
    for prediction in ("fold_mean", "refit_full"):
        raw = _config("diff").model_dump(mode="json", by_alias=True)
        raw.update(test_prediction=prediction, explain={"enabled": True, "max_samples": 30})
        config = ExperimentConfig.model_validate(raw)
        result = run_experiment(
            config, prepare_dataset(config, train, test.drop("y")), output_root=tmp_path
        )
        pred = result.cv_result.test_pred
        assert pred is not None
        # 学習期間の最終値の続きを、元の尺度で正しく外挿できている
        assert pred == pytest.approx(test["y"].to_numpy(), abs=0.05)
        assert (result.output_dir / "horizon_scores.csv").is_file()
        assert (result.output_dir / "shap" / "shap_beeswarm.png").is_file()
