"""1実験（データ読み込み → CV学習 → 予測保存 → ログ記録）の実行。

CLI（`scripts/run_experiment.py`）とNotebookのどちらからも呼べるよう、
データ準備（`prepare_dataset`）と実行（`run_experiment`）を分けている。
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import polars as pl
import yaml

from modeling.config import ExperimentConfig
from modeling.cv import make_folds
from modeling.dataset import Dataset, cross_validate
from modeling.explain import compute_oof_shap, save_shap_outputs
from modeling.forecasting import ForecastCVResult, make_builder
from modeling.io import OOF_FILENAME, TEST_FILENAME, predictions_to_frame, save_predictions
from modeling.tasks import encode_target
from modeling.tracking import NullTracker, Tracker
from modeling.trainer import CVResult
from modeling.tuning import TuningResult, tune
from util.csv_io import read_csv_auto
from util.paths import ensure_parent_dir, get_repo_root, outputs_dir
from util.plotting import add_caption, ensure_japanese_font

__all__ = [
    "Dataset",
    "ExperimentResult",
    "load_dataset",
    "prepare_dataset",
    "run_experiment",
]


@dataclass
class ExperimentResult:
    """実験結果。

    Attributes:
        cv_result: CV学習の結果。
        output_dir: 予測・スコア等を保存したディレクトリ。
        run_id: MLflowのrun ID（記録していなければNone）。
        tuning_result: チューニング結果（チューニングしていなければNone）。
    """

    cv_result: CVResult
    output_dir: Path
    run_id: str | None
    tuning_result: TuningResult | None = None


def resolve_path(path: Path) -> Path:
    """相対パスをリポジトリルート基準の絶対パスに解決する。"""
    return path if path.is_absolute() else get_repo_root() / path


def load_table(path: Path) -> pl.DataFrame:
    """CSV（エンコーディング自動判定）またはParquetを読み込む。

    Raises:
        ValueError: 未対応の拡張子の場合。
    """
    resolved = resolve_path(path)
    if resolved.suffix == ".parquet":
        return pl.read_parquet(resolved)
    if resolved.suffix == ".csv":
        return read_csv_auto(resolved)
    raise ValueError(f"未対応のファイル形式です: {resolved}")


def select_feature_columns(config: ExperimentConfig, columns: list[str]) -> list[str]:
    """特徴量として使う列を決める。

    `data.feature_cols` 指定時はそれを使う。未指定なら目的変数・ID列・`drop_cols` 以外の全列。
    """
    data = config.data
    if data.feature_cols is not None:
        return list(data.feature_cols)
    excluded = {data.target, *data.drop_cols}
    if data.id_col is not None:
        excluded.add(data.id_col)
    return [c for c in columns if c not in excluded]


def prepare_dataset(
    config: ExperimentConfig, train: pl.DataFrame, test: pl.DataFrame | None = None
) -> Dataset:
    """読み込み済みのDataFrameから学習用データ一式を作る。

    `data.time_col` が指定されていれば、学習・テストデータを時刻の昇順に並べ替える
    （時系列CVは行順を時間順とみなすため）。

    Args:
        config: 実験設定。
        train: 学習データ。
        test: テストデータ（任意）。

    Returns:
        学習用データ一式。

    Raises:
        KeyError: 設定で指定した列がデータに無い場合。
    """
    data = config.data
    required = [data.target, data.id_col, data.group_col, data.time_col]
    missing = [c for c in required if c is not None and c not in train.columns]
    if missing:
        raise KeyError(f"学習データに列がありません: {missing}")
    if data.time_col is not None:
        train = train.sort(data.time_col, maintain_order=True)
        if test is not None and data.time_col in test.columns:
            test = test.sort(data.time_col, maintain_order=True)

    features = select_feature_columns(config, train.columns)
    y, classes = encode_target(train[data.target].to_numpy(), config.task)
    groups = None if data.group_col is None else train[data.group_col].to_numpy()
    folds = make_folds(config.cv, train, y, groups, time_column=data.time_col)
    ids_test = None if test is None or data.id_col is None else test[data.id_col]

    if config.forecast is not None:
        # 学習データには真の過去値から目的変数の特徴量を付ける（1期先モデルの学習用）。
        # テストデータの特徴量は予測値を使って再帰的に作るため、ここでは作らない。
        builder = make_builder(config)
        if config.forecast.series_col is not None and config.forecast.series_col not in train:
            raise KeyError(f"学習データに列がありません: {config.forecast.series_col}")
        frame = builder.build(train)
        return Dataset(
            X=frame.select([*features, *builder.feature_names]),
            y=y,
            classes=classes,
            folds=folds,
            ids=None if data.id_col is None else train[data.id_col],
            ids_test=ids_test,
            frame=train,
            test_frame=test,
        )

    return Dataset(
        X=train.select(features),
        y=y,
        classes=classes,
        folds=folds,
        ids=None if data.id_col is None else train[data.id_col],
        X_test=None if test is None else test.select(features),
        ids_test=ids_test,
    )


def load_dataset(config: ExperimentConfig) -> Dataset:
    """設定に従ってファイルを読み込み、学習用データ一式を作る。"""
    train = load_table(config.data.train_path)
    test = None if config.data.test_path is None else load_table(config.data.test_path)
    return prepare_dataset(config, train, test)


def make_output_dir(name: str, root: Path | None = None) -> Path:
    """`outputs/experiments/{実験名}/{実行日時}/` を作って返す。"""
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = (root or outputs_dir() / "experiments") / name / stamp
    out.mkdir(parents=True, exist_ok=False)
    return out


def save_cv_outputs(
    config: ExperimentConfig, dataset: Dataset, result: CVResult, output_dir: Path
) -> list[Path]:
    """設定・スコア・予測をファイルに保存し、保存したパスのリストを返す。"""
    config_path = ensure_parent_dir(output_dir / "config.yaml")
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            config.model_dump(mode="json", by_alias=True) | {"resolved_params": result.params},
            f,
            allow_unicode=True,
            sort_keys=False,
        )

    # fold別スコア（最終行にOOF全体のスコア）
    score_rows: list[dict[str, Any]] = [
        {"fold": str(i), **{m: s[i] for m, s in result.fold_scores.items()}}
        for i in range(len(dataset.folds))
    ]
    score_rows.append({"fold": "oof", **result.oof_scores})
    scores_path = output_dir / "cv_scores.csv"
    pl.DataFrame(score_rows).write_csv(scores_path)

    oof = predictions_to_frame(result.oof_pred, dataset.ids, result.fold_ids, dataset.y)
    paths = [config_path, scores_path, save_predictions(oof, output_dir / OOF_FILENAME)]
    if result.test_pred is not None:
        test = predictions_to_frame(result.test_pred, dataset.ids_test)
        paths.append(save_predictions(test, output_dir / TEST_FILENAME))
    return paths


def save_horizon_outputs(
    config: ExperimentConfig, result: ForecastCVResult, output_dir: Path
) -> list[Path]:
    """再帰予測のステップ別スコア（表と折れ線グラフ）を保存する。"""
    table_path = output_dir / "horizon_scores.csv"
    result.horizon_scores.write_csv(table_path)
    metric = config.primary_metric
    scores = result.horizon_scores.drop_nulls(metric)

    ensure_japanese_font()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.plot(scores["step"].to_list(), scores[metric].to_list(), marker="o", label="再帰予測")
    onestep = result.onestep_oof_scores.get(metric)
    if onestep is not None:
        ax.axhline(onestep, color="gray", linestyle="--", label="1期先予測（真のラグ使用）")
    ax.set_xlabel("予測ステップ（検証期間の何期先か）")
    ax.set_ylabel(metric)
    ax.set_title(f"{config.name}: 予測ステップ別の {metric}")
    ax.legend()
    add_caption(
        fig,
        f"全foldの検証期間をステップ別に集計（各ステップの件数 n は horizon_scores.csv 参照）。"
        f" 再帰OOF {metric}={result.oof_scores[metric]:.6g}",
    )
    figure_path = output_dir / "horizon_error.png"
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [table_path, figure_path]


def run_experiment(
    config: ExperimentConfig,
    dataset: Dataset | None = None,
    tracker: Tracker | None = None,
    output_root: Path | None = None,
    params: dict[str, Any] | None = None,
    tune_params: bool | None = None,
    n_trials: int | None = None,
    optuna_dir: Path | None = None,
    explain: bool | None = None,
) -> ExperimentResult:
    """1実験を実行する。

    チューニングする場合は、同じrunの中で「探索（各試行は子run）→ 最良パラメータでCV学習」
    の順に実行し、最良パラメータでの結果を本runのスコア・予測として記録する。

    Args:
        config: 実験設定。
        dataset: 学習用データ（Noneなら設定に従ってファイルから読み込む）。
        tracker: 実験ログの記録先（Noneなら記録しない）。
        output_root: 出力のルート（Noneなら `outputs/experiments`）。
        params: モデルパラメータ（Noneなら設定値。チューニング時は無視される）。
        tune_params: チューニングするか（Noneなら `config.tuning.enabled`）。
        n_trials: 試行回数の上書き（Noneなら `config.tuning.n_trials`）。
        optuna_dir: Optuna studyを保存するディレクトリ（Noneなら `outputs/optuna`）。
        explain: SHAPを計算するか（Noneなら `config.explain.enabled`）。

    Returns:
        実験結果。
    """
    dataset = dataset or load_dataset(config)
    tracker = tracker or NullTracker()
    output_dir = make_output_dir(config.name, output_root)
    do_tune = config.tuning.enabled if tune_params is None else tune_params

    tags = {"model": config.model.name, "task": str(config.task), "cv": config.cv.method}
    if do_tune:
        tags["tuned"] = "true"
    tuning_result: TuningResult | None = None
    with tracker.start_run(run_name=config.name, tags=tags):
        if do_tune:
            storage = (optuna_dir or outputs_dir() / "optuna") / f"{config.name}.db"
            tuning_result = tune(
                config, dataset, n_trials=n_trials, storage_path=storage, tracker=tracker
            )
            params = tuning_result.best_params
            tracker.log_metrics({f"tuning_best_{config.primary_metric}": tuning_result.best_value})
            trials_path = output_dir / "tuning_trials.csv"
            tuning_result.trials.write_csv(trials_path)
            tracker.log_artifact(trials_path)
        result = cross_validate(config, dataset, params=params)
        tracker.log_params(
            {
                "config": config.model_dump(mode="json", by_alias=True),
                "resolved_params": result.params,
                "n_train": dataset.X.height,
                "n_features": dataset.X.width,
            }
        )
        for i in range(len(dataset.folds)):
            tracker.log_metrics({f"fold_{m}": s[i] for m, s in result.fold_scores.items()}, step=i)
        tracker.log_metrics({f"cv_mean_{k}": v for k, v in result.mean_scores().items()})
        tracker.log_metrics({f"cv_std_{k}": v for k, v in result.std_scores().items()})
        tracker.log_metrics({f"oof_{k}": v for k, v in result.oof_scores.items()})
        for path in save_cv_outputs(config, dataset, result, output_dir):
            tracker.log_artifact(path)
        if isinstance(result, ForecastCVResult):
            # 参考: 真のラグを使う1期先予測のスコア（再帰予測との差が誤差の蓄積分）
            tracker.log_metrics(
                {f"onestep_oof_{k}": v for k, v in result.onestep_oof_scores.items()}
            )
            for path in save_horizon_outputs(config, result, output_dir):
                tracker.log_artifact(path)
        if config.explain.enabled if explain is None else explain:
            shap_result = compute_oof_shap(
                config, dataset.X, dataset.folds, result, n_classes=dataset.n_classes
            )
            class_names = None if dataset.classes is None else [str(c) for c in dataset.classes]
            for path in save_shap_outputs(
                shap_result, output_dir / "shap", config.name, class_names
            ):
                tracker.log_artifact(path, artifact_path="shap")
        run_id = tracker.active_run_id()

    return ExperimentResult(
        cv_result=result, output_dir=output_dir, run_id=run_id, tuning_result=tuning_result
    )
