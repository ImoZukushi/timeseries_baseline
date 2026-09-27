"""evaluation パッケージ（誤差評価の可視化部品）のテスト。"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm
from sklearn import metrics as skm
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, mean_absolute_error
from statsmodels.stats.outliers_influence import OLSInfluence

from eda.time_series_eda import compute_acf
from evaluation.classification import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    scores_to_labels,
)
from evaluation.curves import (
    HorizonErrorDisplay,
    TrainingHistory,
    TrainingHistoryDisplay,
    compute_learning_curve,
    compute_validation_curve,
    most_recent_first,
    plot_learning_curve,
    plot_validation_curve,
)
from evaluation.influence import InfluenceDisplay, compute_influence
from evaluation.residuals import (
    QQPlotDisplay,
    ResidualCorrelogramDisplay,
    ResidualDistributionDisplay,
    ResidualPlotDisplay,
    residual_summary,
)


@pytest.fixture(autouse=True)
def _close_figures() -> Any:
    yield
    plt.close("all")


def _regression_data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n) * 3
    return y, y + rng.normal(size=n)


# --- 残差 -------------------------------------------------------------------------------


def test_residual_summary() -> None:
    summary = residual_summary([3.0, 5.0, 7.0, np.nan], [2.0, 5.0, 9.0, 1.0])
    # 残差は 1, 0, -2（予測が欠損でない行のみ。実測の欠損行は除外）
    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(-1 / 3)
    assert summary["mae"] == pytest.approx(1.0)
    assert summary["rmse"] == pytest.approx(np.sqrt(5 / 3))


def test_residual_displays_have_sklearn_style_attributes() -> None:
    y, p = _regression_data()
    dist = ResidualDistributionDisplay.from_predictions(y, p)
    assert dist.figure_ is not None and dist.ax_ is not None
    assert dist.residuals.size == y.size
    # 既存のaxに描き直せる（sklearnのDisplayと同じ使い方）
    fig, ax = plt.subplots()
    assert dist.plot(ax=ax).ax_ is ax
    fig, axes = plt.subplots(1, 2)
    plot = ResidualPlotDisplay.from_predictions(y, p, ax=axes, max_points=50)
    assert plot.figure_ is fig
    assert len(plot.displays_) == 2


def test_residual_plot_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="長さ"):
        ResidualPlotDisplay.from_predictions([1.0, 2.0], [1.0])


def test_qq_plot_is_nearly_straight_for_normal_residuals() -> None:
    y, p = _regression_data(n=2000)
    qq = QQPlotDisplay.from_predictions(y, p)
    assert qq.r > 0.99
    assert np.all(np.diff(qq.ordered) >= 0)


def test_residual_correlogram_matches_eda_acf_and_handles_groups() -> None:
    rng = np.random.default_rng(0)
    e = np.zeros(300)
    for t in range(1, 300):
        e[t] = 0.7 * e[t - 1] + rng.normal()  # AR(1) の残差
    disp = ResidualCorrelogramDisplay.from_residuals({"A": e, "B": rng.normal(size=300)}, nlags=10)
    expected = compute_acf(pl.Series(e), nlags=10)
    assert expected is not None and disp.acf["A"] == pytest.approx(expected)
    assert disp.acf["A"][1] > 0.5 > abs(disp.acf["B"][1])
    assert np.asarray(disp.ax_).shape == (2, 2)
    # データが少なすぎる系列は「データ不足」と表示して落ちない
    short = ResidualCorrelogramDisplay.from_residuals({"short": [1.0, 2.0]})
    assert short.acf["short"] is None


# --- Leverage / Cook の距離 ---------------------------------------------------------------


def test_influence_matches_statsmodels_for_ols() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(size=80)
    X[5] = [6.0, -6.0, 6.0]  # 外れた特徴量を持つ行
    fit = sm.OLS(y, sm.add_constant(X)).fit()
    expected = OLSInfluence(fit)
    result = compute_influence(X, fit.resid)
    assert result["leverage"].to_numpy() == pytest.approx(expected.hat_matrix_diag)
    assert result["cooks_distance"].to_numpy() == pytest.approx(expected.cooks_distance[0])
    assert result["standardized_residual"].to_numpy() == pytest.approx(
        expected.resid_studentized_internal
    )
    # Leverage の合計 = 特徴量数 + 切片
    assert result["leverage"].sum() == pytest.approx(4.0)
    assert int(result.sort("leverage", descending=True)["row"][0]) == 5


def test_influence_handles_collinearity_and_missing_values() -> None:
    rng = np.random.default_rng(2)
    a = rng.normal(size=50)
    X = np.column_stack([a, 2 * a, rng.normal(size=50)])  # 2列目は1列目の定数倍
    X[3, 2] = np.nan
    result = compute_influence(X, rng.normal(size=50))
    # 線形従属な列はランクに数えない（切片 + 2列）
    assert result["leverage"].sum() == pytest.approx(3.0)
    assert result["cooks_distance"].null_count() == 0
    with pytest.raises(ValueError, match="ランク"):
        compute_influence(rng.normal(size=(3, 5)), rng.normal(size=3))


def test_influence_display_top_and_original_rows() -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 2))
    y = X.sum(axis=1)
    p = y + rng.normal(size=60) * 0.1
    p[10] = np.nan  # 予測が欠損の行は除かれるが、行番号は元のまま
    y[20] += 10  # 大きく外れた行
    disp = InfluenceDisplay.from_predictions(X, y, p)
    assert 10 not in disp.influence["row"].to_list()
    assert int(disp.top(1)["row"][0]) == 20
    assert np.asarray(disp.ax_).size == 2


# --- 分類 -------------------------------------------------------------------------------


def test_confusion_matrix_binary_and_multiclass() -> None:
    y = np.array([0, 0, 1, 1, 1])
    proba = np.array([0.2, 0.6, 0.7, 0.4, 0.9])
    disp = ConfusionMatrixDisplay.from_predictions(y, proba, class_names=["neg", "pos"])
    assert disp.matrix.tolist() == skm.confusion_matrix(y, scores_to_labels(proba)).tolist()
    assert disp.labels == ["neg", "pos"]
    multi = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.3, 0.3, 0.4], [0.5, 0.4, 0.1]])
    disp_m = ConfusionMatrixDisplay.from_predictions([0, 1, 2, 1], multi)
    assert disp_m.matrix.tolist() == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert disp_m.threshold is None


def test_roc_and_pr_match_sklearn_for_binary() -> None:
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 200)
    score = np.clip(y * 0.4 + rng.uniform(size=200) * 0.6, 0, 1)
    roc = RocCurveDisplay.from_predictions(y, score)
    pr = PrecisionRecallDisplay.from_predictions(y, score)
    assert roc.macro == pytest.approx(skm.roc_auc_score(y, score))
    assert pr.macro == pytest.approx(skm.average_precision_score(y, score))
    assert roc.ax_.get_xlabel().startswith("偽陽性率")


def test_multiclass_curves_skip_classes_missing_from_data() -> None:
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 100)  # クラス2は実測に現れない
    proba = rng.dirichlet(np.ones(3), 100)
    roc = RocCurveDisplay.from_predictions(y, proba, class_names=["a", "b", "c"])
    assert set(roc.scores) == {"a", "b"}
    assert roc.macro == pytest.approx(np.mean(list(roc.scores.values())))


# --- 誤差曲線 ---------------------------------------------------------------------------


class _RecordingRegressor(BaseEstimator, RegressorMixin):
    """fitで受け取った行の最小の特徴量値（＝最も古い行）を記録する回帰器。"""

    def fit(self, X: Any, y: Any) -> _RecordingRegressor:
        self.min_seen_ = float(np.min(X))
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), self.mean_)


def test_learning_curve_time_ordered_uses_recent_rows() -> None:
    X = np.arange(100, dtype=float).reshape(-1, 1)  # 値 = 時刻
    y = X.ravel() * 0.1
    folds = [(np.arange(80), np.arange(80, 100))]
    assert most_recent_first(folds)[0][0][0] == 79
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    result = compute_learning_curve(
        _RecordingRegressor(),
        X,
        y,
        folds,
        scorer,
        train_sizes=[0.25, 1.0],
        score_name="mae",
        negate=True,
        time_ordered=True,
    )
    assert result.x.tolist() == [20, 80]
    # 直近の20行で学習した方が、検証期間に近い平均になり誤差が小さい
    assert -result.test_scores[0, 0] < -result.test_scores[1, 0]
    ax = plot_learning_curve(result).ax_
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["学習", "検証"]


def test_validation_curve_shapes_and_plot() -> None:
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(6)
    X = rng.normal(size=(120, 3))
    y = X @ np.array([1.0, 2.0, 3.0]) + rng.normal(size=120)
    folds = [(np.arange(0, 80), np.arange(80, 120)), (np.arange(40, 120), np.arange(0, 40))]
    result = compute_validation_curve(
        Ridge(),
        X,
        y,
        folds,
        "neg_mean_absolute_error",
        param_name="alpha",
        param_range=[0.01, 1.0, 100.0],
        score_name="mae",
        negate=True,
    )
    assert result.train_scores.shape == (3, 2)
    assert plot_validation_curve(result, "alpha").ax_.get_xlabel() == "alpha"


def test_training_history_and_horizon_displays() -> None:
    history = TrainingHistory("l2", [3.0, 2.0, 1.0], [3.5, 2.5, 2.4], best_iteration=3)
    disp = TrainingHistoryDisplay.from_histories([history, TrainingHistory("l2", [1.0], None)])
    # fold 0: 検証・学習・最良反復の縦線、fold 1: 学習のみ → 4本
    assert len(disp.ax_.lines) == 4
    scores = pl.DataFrame({"step": [1, 2, 3], "mae": [1.0, 1.5, None]})
    horizon = HorizonErrorDisplay.from_scores(scores, "mae", reference=0.8)
    assert horizon.steps.tolist() == [1, 2]  # 欠損のステップは除く
    assert horizon.reference == 0.8


@pytest.mark.parametrize("kind", ["distribution", "qq"])
def test_plot_residuals_by_group(kind: str) -> None:
    from evaluation.residuals import plot_residuals_by_group

    groups = {name: _regression_data(seed=i) for i, name in enumerate(["A", "B", "C"])}
    fig, displays = plot_residuals_by_group(kind, groups)  # type: ignore[arg-type]
    assert set(displays) == {"A", "B", "C"}
    # 2列 × 2行のうち3枚を使い、余った1枚は非表示
    assert len(fig.axes) == 4
    assert sum(ax.axison for ax in fig.axes) == 3
    assert displays["B"].ax_.get_title().startswith("B: ")
