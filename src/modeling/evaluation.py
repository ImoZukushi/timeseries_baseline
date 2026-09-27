"""CV結果から誤差評価の図・表を作成する（`evaluation` パッケージの部品を実験につなぐ）。

`run_experiment` から呼ばれ、タスクに応じて次を `{出力}/evaluation/` に保存する。

| タスク | 作る図 |
|---|---|
| 回帰・時系列 | 残差分布・残差プロット・Q-Q・Leverage/Cookの距離・残差のACF/PACF（時刻列あり） |
| 二値分類 | 混同行列（確率0.5以上を陽性）・ROC曲線・PR曲線 |
| 多クラス分類 | 混同行列（確率が最大のクラス）・ROC曲線・PR曲線（One-vs-Rest） |
| 共通 | 学習の推移（記録があれば）・学習曲線／検証曲線（設定で有効な場合） |

すべてOOF（各foldのモデルが、学習に使っていない検証データに対して出した予測）で評価する。
再帰予測の実験では、OOFは元の尺度の再帰予測の値。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import make_scorer

from evaluation.classification import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from evaluation.curves import (
    TrainingHistoryDisplay,
    compute_learning_curve,
    compute_validation_curve,
    plot_learning_curve,
    plot_validation_curve,
)
from evaluation.influence import InfluenceDisplay, compute_influence
from evaluation.residuals import (
    QQPlotDisplay,
    ResidualCorrelogramDisplay,
    ResidualDistributionDisplay,
    ResidualPlotDisplay,
    plot_residuals_by_group,
    residual_summary,
)
from modeling.config import ExperimentConfig
from modeling.dataset import Dataset
from modeling.metrics import get_metric
from modeling.models import get_model_spec
from modeling.pipeline import MODEL_STEP, build_pipeline
from modeling.tasks import is_classification
from modeling.trainer import CVResult
from util.plotting import add_caption

EVALUATION_DIR = "evaluation"

# Leverage の計算（特異値分解）に使う最大行数。多い場合はfoldごとに無作為に抽出する
_MAX_INFLUENCE_ROWS = 200_000


def save_evaluation_outputs(
    config: ExperimentConfig, dataset: Dataset, cv_result: CVResult, output_dir: Path
) -> list[Path]:
    """タスクに応じた誤差評価の図・表を保存し、保存したパスのリストを返す。

    Args:
        config: 実験設定。
        dataset: 学習用データ一式（`run_cv` / 再帰バックテストに渡したもの）。
        cv_result: CV結果（OOF予測・foldモデル）。
        output_dir: 実験の出力ディレクトリ（この下の `evaluation/` に保存する）。

    Returns:
        保存したファイルのパス。
    """
    out = output_dir / EVALUATION_DIR
    out.mkdir(parents=True, exist_ok=True)
    saver = _Saver(out, config.name)
    rows = _predicted_rows(cv_result)
    y, pred = dataset.y[rows], cv_result.oof_pred[rows]
    note = f"OOF予測（各foldのモデルが学習に使っていない検証データへの予測）n={rows.size:,}"

    if is_classification(config.task):
        names = None if dataset.classes is None else [str(c) for c in dataset.classes]
        saver.figure(
            ConfusionMatrixDisplay.from_predictions(y, pred, class_names=names).figure_,
            "confusion_matrix.png",
            note,
        )
        saver.figure(
            RocCurveDisplay.from_predictions(y, pred, class_names=names).figure_,
            "roc_curve.png",
            note,
        )
        saver.figure(
            PrecisionRecallDisplay.from_predictions(y, pred, class_names=names).figure_,
            "pr_curve.png",
            note,
        )
    else:
        _save_regression(config, dataset, cv_result, rows, saver, note)

    _save_training_history(config, cv_result, len(dataset.folds), saver)
    if config.evaluation.learning_curve.enabled or config.evaluation.validation_curve.param:
        _save_curves(config, dataset, cv_result, saver)
    return saver.paths


class _Saver:
    """図・表を保存し、保存したパスを記録する。"""

    def __init__(self, directory: Path, experiment_name: str) -> None:
        self.directory = directory
        self.experiment_name = experiment_name
        self.paths: list[Path] = []

    def figure(self, fig: plt.Figure, filename: str, note: str) -> None:
        add_caption(fig, f"{self.experiment_name}: {note}")
        path = self.directory / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.paths.append(path)

    def table(self, frame: pl.DataFrame, filename: str) -> None:
        path = self.directory / filename
        frame.write_csv(path)
        self.paths.append(path)


def _predicted_rows(cv_result: CVResult) -> np.ndarray:
    """OOF予測がある行（どのfoldの検証にも入らない行・予測が欠損の行を除く）。"""
    pred = cv_result.oof_pred
    finite = np.isfinite(pred if pred.ndim == 1 else pred.sum(axis=1))
    return np.flatnonzero((cv_result.fold_ids >= 0) & finite)


# --- 回帰・時系列 ----------------------------------------------------------------------


def _save_regression(
    config: ExperimentConfig,
    dataset: Dataset,
    cv_result: CVResult,
    rows: np.ndarray,
    saver: _Saver,
    note: str,
) -> None:
    """回帰・時系列の残差の図・表を保存する。"""
    ev = config.evaluation
    y, pred = dataset.y[rows], cv_result.oof_pred[rows]
    series_rows = _series_rows(config, dataset, rows)

    # 残差の要約: 全体の1行（series=all）＋ 複数系列なら系列ごとの行
    summary_rows = [{"series": "all", **residual_summary(y, pred)}]
    if series_rows is not None:
        summary_rows += [
            {"series": name, **residual_summary(dataset.y[r], cv_result.oof_pred[r])}
            for name, r in series_rows.items()
        ]
    saver.table(pl.DataFrame(summary_rows), "residual_summary.csv")

    if series_rows is None:
        saver.figure(
            ResidualDistributionDisplay.from_predictions(y, pred).figure_,
            "residual_distribution.png",
            note,
        )
        saver.figure(QQPlotDisplay.from_predictions(y, pred).figure_, "qq_plot.png", note)
    else:
        # 尺度の違う系列の残差を1つにまとめると分布の形が読めないため、系列ごとに分けて描く
        shown = dict(list(series_rows.items())[: ev.max_series])
        groups = {name: (dataset.y[r], cv_result.oof_pred[r]) for name, r in shown.items()}
        shown_note = _series_note(note, len(shown), len(series_rows))
        fig, _ = plot_residuals_by_group("distribution", groups)
        saver.figure(fig, "residual_distribution.png", shown_note)
        fig, _ = plot_residuals_by_group("qq", groups)
        saver.figure(fig, "qq_plot.png", shown_note)
    saver.figure(
        ResidualPlotDisplay.from_predictions(y, pred, max_points=ev.max_points).figure_,
        "residual_plot.png",
        note,
    )

    influence, n_params = _oof_influence(dataset, cv_result, rows)
    if influence is not None:
        display = InfluenceDisplay(influence, n_params).plot(max_points=ev.max_points)
        saver.figure(
            display.figure_,
            "leverage_cooks_distance.png",
            f"{note}。Leverageは各foldのモデルに入力した特徴量（前処理後）から計算した"
            "線形回帰の診断量で、非線形モデルでは影響の大きいサンプルの目安",
        )
        top = display.top(20)
        if dataset.ids is not None:
            top = top.with_columns(dataset.ids.gather(top["row"].to_numpy()).alias("id"))
        saver.table(top, "top_cooks_distance.csv")

    if config.data.time_col is not None:
        # 学習データは時刻順に並べ替え済みなので、行番号の順がそのまま時刻順
        if series_rows is None:
            residuals: dict[str, np.ndarray] = {"残差": y - pred}
            acf_note = note
        else:
            shown_rows = list(series_rows.items())[: ev.max_series]
            residuals = {name: dataset.y[r] - cv_result.oof_pred[r] for name, r in shown_rows}
            acf_note = _series_note(note, len(shown_rows), len(series_rows))
        saver.figure(
            ResidualCorrelogramDisplay.from_residuals(residuals).figure_,
            "residual_acf_pacf.png",
            f"{acf_note}。時刻順に並べた残差で計算（点線は無相関の場合の95%信頼区間）",
        )


def _oof_influence(
    dataset: Dataset, cv_result: CVResult, rows: np.ndarray
) -> tuple[pl.DataFrame | None, int]:
    """foldごとに、各foldのモデルの入力（前処理後の検証データ）で Leverage と Cookの距離を求める。

    前処理（ワンホットのカテゴリなど）はfoldごとに学習されるため、列がfoldで揃うとは限らない。
    そのためfoldごとに計算して連結する（OOF予測と同じ考え方）。
    """
    predicted = np.zeros(dataset.X.height, dtype=bool)
    predicted[rows] = True
    frames, ranks = [], []
    rng = np.random.default_rng(0)
    for fold in range(len(dataset.folds)):
        fold_rows = np.flatnonzero(predicted & (cv_result.fold_ids == fold))
        if fold_rows.size == 0 or fold >= len(cv_result.models):
            continue
        if fold_rows.size > _MAX_INFLUENCE_ROWS:
            fold_rows = np.sort(rng.choice(fold_rows, _MAX_INFLUENCE_ROWS, replace=False))
        model_input = cv_result.models[fold][:-1].transform(dataset.X[fold_rows])
        X = np.asarray(model_input, dtype=np.float64)
        residuals = dataset.y[fold_rows] - cv_result.oof_pred[fold_rows]
        try:
            inf = compute_influence(X, residuals)
        except ValueError:
            # 行数が特徴量数以下のfoldは計算できないため飛ばす
            continue
        ranks.append(float(inf["leverage"].sum()))
        frames.append(inf.with_columns(pl.Series("row", fold_rows), pl.lit(fold).alias("fold")))
    if not frames:
        return None, 0
    return pl.concat(frames), int(round(float(np.mean(ranks))))


def _series_rows(
    config: ExperimentConfig, dataset: Dataset, rows: np.ndarray
) -> dict[str, np.ndarray] | None:
    """複数系列（`forecast.series_col`）の場合、系列名 → その系列の行番号（元の並び順）を返す。

    単一系列の場合はNone。系列名の昇順に並べる。
    """
    series_col = config.forecast.series_col if config.forecast is not None else None
    if series_col is None or dataset.frame is None:
        return None
    ids = dataset.frame[series_col].to_numpy()[rows].astype(str)
    return {name: rows[ids == name] for name in sorted(set(ids))}


def _series_note(note: str, n_shown: int, n_total: int) -> str:
    """系列ごとに描いた図の注記（系列数を絞った場合はその旨を加える）。"""
    if n_shown < n_total:
        return (
            f"{note}。系列ごとに表示（全{n_total}系列のうち名前順の先頭{n_shown}系列。"
            "evaluation.max_series で変更可）"
        )
    return f"{note}。系列ごとに表示"


# --- 学習の推移・学習曲線・検証曲線 ------------------------------------------------------


def _save_training_history(
    config: ExperimentConfig, cv_result: CVResult, n_folds: int, saver: _Saver
) -> None:
    """foldモデルの学習の推移（反復ごとの損失）を描く（記録があるモデルのみ）。"""
    if not config.evaluation.training_history:
        return
    spec = get_model_spec(config.model.name)
    histories = [
        h
        for model in cv_result.models[:n_folds]
        if (h := spec.training_history(model.named_steps[MODEL_STEP])) is not None
    ]
    if histories:
        saver.figure(
            TrainingHistoryDisplay.from_histories(histories).figure_,
            "training_history.png",
            "各foldのモデルの反復ごとの損失（モデル内部の指標。検証は各foldの検証データ）",
        )


def _save_curves(
    config: ExperimentConfig, dataset: Dataset, cv_result: CVResult, saver: _Saver
) -> None:
    """学習曲線・検証曲線を計算して描く（foldごとに学習し直すため時間がかかる）。"""
    ev = config.evaluation
    estimator, X, y, folds, scorer, negate, scale_note = _curve_inputs(config, dataset, cv_result)
    metric = config.primary_metric
    common = "各foldで学習し直した学習・検証スコア（early stoppingなし）。" + scale_note
    if ev.learning_curve.enabled:
        time_ordered = config.data.time_col is not None
        result = compute_learning_curve(
            estimator,
            X,
            y,
            folds,
            scorer,
            train_sizes=ev.learning_curve.train_sizes,
            score_name=metric,
            negate=negate,
            time_ordered=time_ordered,
        )
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        plot_learning_curve(result, ax=ax)
        order = "直近のデータから指定量を使用。" if time_ordered else ""
        saver.figure(fig, "learning_curve.png", common + order)
        saver.table(_curve_table(result, "train_size"), "learning_curve.csv")
    if ev.validation_curve.param:
        param = ev.validation_curve.param
        name = param if param.startswith(f"{MODEL_STEP}__") else f"{MODEL_STEP}__{param}"
        result = compute_validation_curve(
            estimator,
            X,
            y,
            folds,
            scorer,
            param_name=name,
            param_range=ev.validation_curve.values,
            score_name=metric,
            negate=negate,
        )
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        plot_validation_curve(result, param, ax=ax)
        saver.figure(fig, "validation_curve.png", common)
        saver.table(_curve_table(result, "param_value"), "validation_curve.csv")


def _curve_inputs(
    config: ExperimentConfig, dataset: Dataset, cv_result: CVResult
) -> tuple[Any, Any, np.ndarray, list[tuple[np.ndarray, np.ndarray]], Any, bool, str]:
    """学習曲線・検証曲線の計算に使う estimator・データ・fold・scorer を用意する。"""
    run_config = config.model_copy(
        update={"model": config.model.model_copy(update={"early_stopping_rounds": None})}
    )
    estimator = build_pipeline(run_config, cv_result.params)
    tt = dataset.target_transform
    # 目的変数を変換している場合、モデルが学習するのは変換後の値（差分が無い先頭行は除く）
    y = dataset.y if tt is None else tt.y_model
    usable = np.isfinite(y.astype(np.float64))
    folds = [(tr[usable[tr]], va[usable[va]]) for tr, va in dataset.folds]
    metric = get_metric(config.primary_metric)
    response = "predict_proba" if is_classification(config.task) else "predict"
    scorer = make_scorer(
        metric.func, greater_is_better=metric.greater_is_better, response_method=response
    )
    notes = []
    if config.forecast is not None:
        notes.append("再帰予測の実験でも、ここでは1期先予測のスコア。")
    if tt is not None:
        notes.append(f"目的変数は変換後（{tt.kind}）の尺度。")
    return estimator, dataset.X, y, folds, scorer, not metric.greater_is_better, "".join(notes)


def _curve_table(result: Any, x_name: str) -> pl.DataFrame:
    """学習曲線・検証曲線の計算結果を表にする（スコアは元の向きに戻す）。"""
    sign = -1.0 if result.negate else 1.0
    return pl.DataFrame(
        {
            x_name: [str(v) for v in result.x],
            "train_mean": sign * result.train_scores.mean(axis=1),
            "train_std": result.train_scores.std(axis=1),
            "valid_mean": sign * result.test_scores.mean(axis=1),
            "valid_std": result.test_scores.std(axis=1),
        }
    )
