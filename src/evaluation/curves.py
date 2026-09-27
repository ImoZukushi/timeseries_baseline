"""誤差曲線（学習曲線・検証曲線・学習の推移・予測ステップ別の誤差）。

- **学習曲線**（データ量）: 学習データの量を増やしたときの学習・検証スコア。
  学習と検証の差が大きければ過学習、両方悪ければ学習不足・特徴量不足、データを増やすと
  検証スコアがまだ伸びるならデータ追加が有効、と読む。sklearn の `learning_curve` を使う。
- **検証曲線**（ハイパーパラメータ）: 1つのパラメータを動かしたときの学習・検証スコア。
  sklearn の `validation_curve` を使う。
- **学習の推移**（反復）: 木の本数・エポックごとの学習・検証の損失。early stopping の妥当性を見る。
- **予測ステップ別の誤差**: 再帰予測で何期先まで誤差がどう増えるか。

学習曲線・検証曲線の `cv` には、`(学習の行番号, 検証の行番号)` の組のリストを渡せる
（時系列なら時間順を守った分割をそのまま使う）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import polars as pl
from sklearn.model_selection import (
    LearningCurveDisplay,
    ValidationCurveDisplay,
    learning_curve,
    validation_curve,
)

from evaluation._common import new_axes
from util.plotting import ensure_japanese_font

Fold = tuple[np.ndarray, np.ndarray]


def most_recent_first(folds: Sequence[Fold]) -> list[Fold]:
    """各foldの学習行の並びを新しい順にする（時系列の学習曲線用）。

    sklearn の `learning_curve` は学習行の先頭から指定量を使うため、並びを逆にすると
    「直近のデータから指定量」（古い側を削る）になる。時系列では予測時点に近いデータほど
    重要なので、こちらの方が実際の使い方に近い。
    """
    return [(np.asarray(tr)[::-1], np.asarray(va)) for tr, va in folds]


@dataclass
class CurveResult:
    """学習曲線・検証曲線の計算結果（sklearn の戻り値をまとめたもの）。

    Attributes:
        x: 横軸（学習曲線は学習データの行数、検証曲線はパラメータの値）。
        train_scores: 学習スコア (len(x), fold数)。
        test_scores: 検証スコア (len(x), fold数)。
        score_name: スコアの名前。
        negate: スコアが符号反転されているか（sklearnの「大きいほど良い」規約のため、
            小さいほど良い指標は負の値で返る。描画時に元に戻す）。
    """

    x: np.ndarray
    train_scores: np.ndarray
    test_scores: np.ndarray
    score_name: str
    negate: bool


def compute_learning_curve(
    estimator: Any,
    X: Any,
    y: Any,
    cv: Sequence[Fold],
    scoring: Any,
    *,
    train_sizes: Sequence[float] = (0.2, 0.4, 0.6, 0.8, 1.0),
    score_name: str = "score",
    negate: bool = False,
    time_ordered: bool = False,
) -> CurveResult:
    """学習曲線を計算する（sklearn の `learning_curve`）。

    Args:
        estimator: sklearn互換のestimator（未学習。foldごとに複製して学習される）。
        X: 特徴量。
        y: 目的変数。
        cv: `(学習の行番号, 検証の行番号)` のリスト。
        scoring: sklearn のscorer（`make_scorer` の結果など）。
        train_sizes: 各foldの学習データのうち使う割合。
        score_name: スコアの名前（図の軸ラベル）。
        negate: scorer が符号反転したスコアを返すか（小さいほど良い指標）。
        time_ordered: Trueなら直近のデータから指定量を使う（時系列向け）。

    Returns:
        計算結果。
    """
    folds = most_recent_first(cv) if time_ordered else list(cv)
    sizes, train, test = learning_curve(
        estimator, X, y, cv=folds, scoring=scoring, train_sizes=list(train_sizes), shuffle=False
    )
    return CurveResult(np.asarray(sizes), train, test, score_name, negate)


def compute_validation_curve(
    estimator: Any,
    X: Any,
    y: Any,
    cv: Sequence[Fold],
    scoring: Any,
    *,
    param_name: str,
    param_range: Sequence[Any],
    score_name: str = "score",
    negate: bool = False,
) -> CurveResult:
    """検証曲線を計算する（sklearn の `validation_curve`）。

    Args:
        estimator: sklearn互換のestimator。
        X: 特徴量。
        y: 目的変数。
        cv: `(学習の行番号, 検証の行番号)` のリスト。
        scoring: sklearn のscorer。
        param_name: 動かすパラメータ（`estimator.set_params` に渡す名前。
            Pipelineなら `model__...`）。
        param_range: パラメータの値のリスト。
        score_name: スコアの名前。
        negate: scorer が符号反転したスコアを返すか。

    Returns:
        計算結果。
    """
    train, test = validation_curve(
        estimator,
        X,
        y,
        param_name=param_name,
        param_range=list(param_range),
        cv=list(cv),
        scoring=scoring,
    )
    return CurveResult(np.asarray(list(param_range)), train, test, score_name, negate)


def plot_learning_curve(result: CurveResult, ax: Any = None, *, title: str = "") -> Any:
    """学習曲線を描く（sklearn の `LearningCurveDisplay`）。"""
    ensure_japanese_font()
    _, ax = new_axes(ax, figsize=(8, 5))
    disp = LearningCurveDisplay(
        train_sizes=result.x,
        train_scores=result.train_scores,
        test_scores=result.test_scores,
        score_name=result.score_name,
    ).plot(ax=ax, negate_score=result.negate, std_display_style="fill_between")
    ax.set_xlabel("学習データの行数（各foldの学習データのうち使った量）")
    ax.set_ylabel(result.score_name)
    ax.set_title(title or f"学習曲線（{result.score_name}、fold間の平均±標準偏差）")
    _relabel(ax)
    return disp


def plot_validation_curve(
    result: CurveResult, param_name: str, ax: Any = None, *, title: str = ""
) -> Any:
    """検証曲線を描く（sklearn の `ValidationCurveDisplay`）。"""
    ensure_japanese_font()
    _, ax = new_axes(ax, figsize=(8, 5))
    numeric = np.issubdtype(result.x.dtype, np.number)
    disp = ValidationCurveDisplay(
        param_name=param_name,
        param_range=result.x if numeric else np.arange(result.x.size),
        train_scores=result.train_scores,
        test_scores=result.test_scores,
        score_name=result.score_name,
    ).plot(ax=ax, negate_score=result.negate, std_display_style="fill_between")
    if not numeric:
        ax.set_xticks(np.arange(result.x.size), [str(v) for v in result.x])
    ax.set_xlabel(param_name)
    ax.set_ylabel(result.score_name)
    ax.set_title(title or f"検証曲線（{param_name} を動かしたときの {result.score_name}）")
    _relabel(ax)
    return disp


def _relabel(ax: Any) -> None:
    """sklearn の凡例（Train / Test）を日本語にする。"""
    handles, labels = ax.get_legend_handles_labels()
    names = {"Train": "学習", "Test": "検証"}
    ax.legend(handles, [names.get(label, label) for label in labels])


@dataclass
class TrainingHistory:
    """1モデル分の学習の推移。

    Attributes:
        metric: 損失の名前（モデル内部の指標。例: `l2`, `logloss`, `loss`）。
        train: 反復ごとの学習データの損失（記録が無ければNone）。
        valid: 反復ごとの検証データの損失（記録が無ければNone）。
        best_iteration: early stopping で選ばれた反復数（1始まり。無ければNone）。
    """

    metric: str
    train: list[float] | None
    valid: list[float] | None
    best_iteration: int | None = None


class TrainingHistoryDisplay:
    """反復（木の本数・エポック）ごとの学習・検証の損失（foldごとに1組の線）。

    実線が検証、破線が学習。縦の点線は early stopping で選ばれた反復数。

    Attributes:
        histories: foldごとの学習の推移。
        figure_: 描画したFigure。
        ax_: 描画したAxes。
    """

    def __init__(self, histories: Sequence[TrainingHistory]) -> None:
        self.histories = list(histories)

    @classmethod
    def from_histories(cls, histories: Sequence[TrainingHistory], *, ax: Any = None) -> Self:
        """学習の推移のリストから描く。"""
        return cls(histories).plot(ax=ax)

    def plot(self, ax: Any = None) -> Self:
        """学習の推移を描く。"""
        ensure_japanese_font()
        fig, ax = new_axes(ax, figsize=(9, 5))
        colors = [f"C{i}" for i in range(10)]
        for i, h in enumerate(self.histories):
            color = colors[i % len(colors)]
            if h.valid is not None:
                ax.plot(
                    np.arange(1, len(h.valid) + 1), h.valid, color=color, label=f"fold {i} 検証"
                )
            if h.train is not None:
                ax.plot(
                    np.arange(1, len(h.train) + 1),
                    h.train,
                    color=color,
                    linestyle="--",
                    alpha=0.7,
                    label=f"fold {i} 学習",
                )
            if h.best_iteration is not None:
                ax.axvline(h.best_iteration, color=color, linestyle=":", linewidth=1)
        metric = self.histories[0].metric if self.histories else ""
        ax.set_xlabel("反復（木の本数・エポック）")
        ax.set_ylabel(f"損失（{metric}）")
        ax.set_title("学習の推移（実線: 検証, 破線: 学習, 点線: early stoppingで選ばれた反復）")
        ax.legend(fontsize=8, ncols=2)
        self.figure_, self.ax_ = fig, ax
        return self


class HorizonErrorDisplay:
    """再帰予測の予測ステップ別の誤差（何期先まで誤差がどう増えるか）。

    Attributes:
        steps: 予測ステップ（1始まり）。
        scores: ステップごとの指標の値。
        metric: 指標の名前。
        reference: 比較用の水準（例: 真のラグを使った1期先予測のスコア）。
        figure_: 描画したFigure。
        ax_: 描画したAxes。
    """

    def __init__(
        self, steps: np.ndarray, scores: np.ndarray, metric: str, reference: float | None
    ) -> None:
        self.steps = steps
        self.scores = scores
        self.metric = metric
        self.reference = reference

    @classmethod
    def from_scores(
        cls,
        horizon_scores: pl.DataFrame,
        metric: str,
        *,
        reference: float | None = None,
        ax: Any = None,
        title: str = "",
    ) -> Self:
        """ステップ別スコアの表（列 `step` と指標名）から描く。"""
        valid = horizon_scores.drop_nulls(metric)
        disp = cls(valid["step"].to_numpy(), valid[metric].to_numpy(), metric, reference)
        return disp.plot(ax=ax, title=title)

    def plot(self, ax: Any = None, *, title: str = "") -> Self:
        """予測ステップ別の誤差を描く。"""
        ensure_japanese_font()
        fig, ax = new_axes(ax, figsize=(10, 5))
        ax.plot(self.steps, self.scores, marker="o", markersize=3, label="再帰予測")
        if self.reference is not None:
            ax.axhline(
                self.reference, color="gray", linestyle="--", label="1期先予測（真のラグ使用）"
            )
        ax.set_xlabel("予測ステップ（検証期間の何期先か）")
        ax.set_ylabel(self.metric)
        ax.set_title(title or f"予測ステップ別の {self.metric}")
        ax.legend()
        self.figure_, self.ax_ = fig, ax
        return self
