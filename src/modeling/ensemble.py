"""複数実験の予測を統合するアンサンブル。

各実験が保存した **OOF予測・テスト予測** を入力にするため、モデルの再学習は不要。
統合方法:

- `mean`: 単純平均
- `rank_mean`: 順位の平均（二値分類のAUC向け）
- `weighted`: OOFで主指標が最良になる重み（非負・合計1）を探索
- `stacking`: 構成要素の予測を特徴量にしたメタモデル

`weighted` / `stacking` は重み・メタモデルをOOFで学習するため、同じOOFで評価すると
楽観的なスコアになる。そこで「fold k 以外のOOFで学習し fold k を予測する」操作を
全foldで行ったアンサンブルのOOFでスコアを計算する（構成要素と同じCV分割を再利用）。

**前提**: 全構成要素が同じ学習データ・同じCV分割で作られていること。
行数・fold番号・目的変数が一致しない場合はエラーにする。
"""

from __future__ import annotations

import datetime as dt
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import polars as pl
import yaml
from scipy.optimize import minimize
from scipy.stats import rankdata

from modeling.config import EnsembleConfig, EnsembleMemberConfig
from modeling.io import (
    OOF_FILENAME,
    TEST_FILENAME,
    frame_to_predictions,
    predictions_to_frame,
    save_predictions,
)
from modeling.metrics import Metric, get_metric
from modeling.models import get_model_spec
from modeling.tasks import Task, predict
from modeling.tracking import NullTracker, Tracker, default_tracking_uri
from util.paths import get_repo_root, outputs_dir


@dataclass
class MemberPredictions:
    """1構成要素分の予測。

    Attributes:
        name: 表示名。
        oof: OOF予測（どのfoldにも入らない行はNaN）。
        fold_ids: 各行のfold番号（-1はどのfoldにも入らない行）。
        target: エンコード済みの目的変数。
        test: テスト予測（無ければNone）。
        test_ids: テストデータの行ID（無ければNone）。
        source: 読み込み元（ディレクトリまたはrun ID）。
    """

    name: str
    oof: np.ndarray
    fold_ids: np.ndarray
    target: np.ndarray
    test: np.ndarray | None
    test_ids: pl.Series | None
    source: str


def latest_run_dir(experiment: str, experiments_root: Path | None = None) -> Path:
    """`outputs/experiments/{実験名}/` 配下で最も新しい実行ディレクトリを返す。

    Raises:
        FileNotFoundError: 実行結果が1つも無い場合。
    """
    root = (experiments_root or outputs_dir() / "experiments") / experiment
    runs = sorted(p for p in root.glob("*") if (p / OOF_FILENAME).is_file())
    if not runs:
        raise FileNotFoundError(f"実験 {experiment} の実行結果が見つかりません: {root}")
    # ディレクトリ名は実行日時（YYYYmmdd_HHMMSS_ffffff）なので名前順＝時刻順
    return runs[-1]


def _download_run_artifacts(run_id: str, tracking_uri: str | None, dst: Path) -> Path:
    """MLflowのrunから予測ファイルをダウンロードしたディレクトリを返す。"""
    import mlflow
    from mlflow.exceptions import MlflowException

    uri = tracking_uri or default_tracking_uri()
    for filename in (OOF_FILENAME, TEST_FILENAME):
        try:
            mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=filename, dst_path=str(dst), tracking_uri=uri
            )
        except (MlflowException, OSError) as e:
            # テスト予測はテストデータ無しの実験では存在しないため、OOFのみ必須とする
            if filename == OOF_FILENAME:
                raise FileNotFoundError(f"run {run_id} に {filename} がありません") from e
    return dst


def load_member(
    member: EnsembleMemberConfig,
    experiments_root: Path | None = None,
    tracking_uri: str | None = None,
) -> MemberPredictions:
    """設定に従って1構成要素分の予測を読み込む。"""
    with tempfile.TemporaryDirectory() as tmp:
        if member.run_id is not None:
            directory = _download_run_artifacts(member.run_id, tracking_uri, Path(tmp))
            source = f"mlflow:{member.run_id}"
        elif member.experiment is not None:
            directory = latest_run_dir(member.experiment, experiments_root)
            source = str(directory)
        else:
            assert member.path is not None
            directory = member.path if member.path.is_absolute() else get_repo_root() / member.path
            source = str(directory)
        oof_frame = pl.read_parquet(directory / OOF_FILENAME)
        test_path = directory / TEST_FILENAME
        test_frame = pl.read_parquet(test_path) if test_path.is_file() else None

    return MemberPredictions(
        name=member.name,
        oof=frame_to_predictions(oof_frame),
        fold_ids=oof_frame["fold"].to_numpy(),
        target=oof_frame["target"].to_numpy(),
        test=None if test_frame is None else frame_to_predictions(test_frame),
        test_ids=None if test_frame is None or "id" not in test_frame.columns else test_frame["id"],
        source=source,
    )


def check_members_consistent(members: list[MemberPredictions]) -> None:
    """構成要素が同じデータ・同じCV分割で作られているかを確認する。

    Raises:
        ValueError: 行数・fold番号・目的変数・予測の形・テスト予測の有無が一致しない場合。
    """
    ref = members[0]
    for m in members[1:]:
        if m.oof.shape != ref.oof.shape:
            raise ValueError(
                f"{m.name} のOOF予測の形 {m.oof.shape} が {ref.oof.shape} と異なります"
            )
        if not np.array_equal(m.fold_ids, ref.fold_ids):
            # CV分割が違うとOOF同士を行ごとに比較・統合できない
            raise ValueError(f"{m.name} のCV分割が {ref.name} と異なります（cv設定を揃える）")
        if not np.array_equal(m.target, ref.target):
            raise ValueError(f"{m.name} の目的変数が {ref.name} と異なります")
        if (m.test is None) != (ref.test is None):
            raise ValueError(f"{m.name} と {ref.name} でテスト予測の有無が異なります")
        if m.test is not None and ref.test is not None and m.test.shape != ref.test.shape:
            raise ValueError(f"{m.name} のテスト予測の形が {ref.name} と異なります")


# --- 統合方法 ------------------------------------------------------------------


class Blender(Protocol):
    """構成要素の予測（`(n_members, n_samples[, n_classes])`）を1つに統合する。"""

    def fit(self, preds: np.ndarray, y: np.ndarray) -> Blender:
        """統合方法を学習する（学習不要な方法では何もしない）。"""
        ...

    def predict(self, preds: np.ndarray) -> np.ndarray:
        """統合した予測を返す。"""
        ...


class MeanBlender:
    """単純平均。"""

    def fit(self, preds: np.ndarray, y: np.ndarray) -> MeanBlender:
        """学習不要。"""
        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        """構成要素の平均。"""
        return np.asarray(preds.mean(axis=0))


class RankMeanBlender:
    """順位の平均（0〜1に正規化）。予測するデータ内での順位を使うため学習不要。"""

    def fit(self, preds: np.ndarray, y: np.ndarray) -> RankMeanBlender:
        """学習不要。"""
        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        """各構成要素の予測を順位に変換して平均する。"""
        ranks = np.stack([(rankdata(p) - 1) / max(len(p) - 1, 1) for p in preds])
        return np.asarray(ranks.mean(axis=0))


class WeightedBlender:
    """主指標が最良になる非負・合計1の重みを探索する加重平均。

    重みは softmax(z) でパラメータ化し、Nelder-Mead法で探索する
    （AUC等の微分できない指標にも使えるようにするため）。

    Args:
        metric: 最適化する指標。
    """

    def __init__(self, metric: Metric) -> None:
        self.metric = metric
        self.weights_: np.ndarray | None = None

    def fit(self, preds: np.ndarray, y: np.ndarray) -> WeightedBlender:
        """OOFで重みを探索する。"""
        sign = -1.0 if self.metric.greater_is_better else 1.0

        def objective(z: np.ndarray) -> float:
            return sign * self.metric(y, _weighted_sum(preds, _softmax(z)))

        n = preds.shape[0]
        result = minimize(
            objective, np.zeros(n), method="Nelder-Mead", options={"maxiter": 200 * n}
        )
        self.weights_ = _softmax(result.x)
        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        """学習済みの重みで加重平均する。"""
        if self.weights_ is None:
            raise RuntimeError("fit を先に呼んでください")
        return _weighted_sum(preds, self.weights_)


class StackingBlender:
    """構成要素の予測を特徴量にしたメタモデル。

    Args:
        model_name: メタモデルの登録名（`modeling.models`）。
        params: メタモデルのパラメータ。
        task: 予測タスク。
        seed: 乱数シード。
        n_classes: 多クラス分類のクラス数。
    """

    def __init__(
        self,
        model_name: str,
        params: dict[str, Any],
        task: Task,
        seed: int,
        n_classes: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.params = params
        self.task = task
        self.seed = seed
        self.n_classes = n_classes
        self.model_: Any = None

    def _features(self, preds: np.ndarray) -> pd.DataFrame:
        """(n_members, n_samples[, n_classes]) をメタモデルの入力表に変換する。"""
        if preds.ndim == 3:
            columns = {
                f"m{i}_c{k}": preds[i, :, k]
                for i in range(preds.shape[0])
                for k in range(preds.shape[2])
            }
        else:
            columns = {f"m{i}": preds[i] for i in range(preds.shape[0])}
        return pd.DataFrame(columns)

    def fit(self, preds: np.ndarray, y: np.ndarray) -> StackingBlender:
        """メタモデルを学習する。"""
        spec = get_model_spec(self.model_name)
        self.model_ = spec.build(self.task, self.params, seed=self.seed)
        self.model_.fit(self._features(preds), y)
        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        """メタモデルで予測する。"""
        return predict(self.model_, self._features(preds), self.task, self.n_classes)


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return np.asarray(e / e.sum())


def _weighted_sum(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """(n_members, ...) の予測を重み付きで合計する。"""
    return np.asarray(np.tensordot(weights, preds, axes=1))


def make_blender(config: EnsembleConfig, n_classes: int | None = None) -> Blender:
    """設定から統合方法を作る。"""
    if config.method == "mean":
        return MeanBlender()
    if config.method == "rank_mean":
        return RankMeanBlender()
    if config.method == "weighted":
        return WeightedBlender(get_metric(config.primary_metric))
    return StackingBlender(
        config.stacking.model.name,
        config.stacking.model.params,
        config.task,
        config.seed,
        n_classes,
    )


# --- 実行 ------------------------------------------------------------------------


@dataclass
class EnsembleResult:
    """アンサンブル結果。

    Attributes:
        oof_pred: アンサンブルのOOF予測（fold毎に他foldで学習した統合方法で予測）。
        test_pred: アンサンブルのテスト予測（全OOFで学習した統合方法で予測）。
        scores: 構成要素とアンサンブルのOOFスコア表。
        weights: 構成要素の重み（`mean` / `weighted` のみ。それ以外はNone）。
        output_dir: 出力ディレクトリ。
        run_id: MLflowのrun ID。
    """

    oof_pred: np.ndarray
    test_pred: np.ndarray | None
    scores: pl.DataFrame
    weights: dict[str, float] | None
    output_dir: Path
    run_id: str | None


def cross_fit_blend(
    blender_factory: Callable[[], Blender],
    preds: np.ndarray,
    y: np.ndarray,
    fold_ids: np.ndarray,
) -> np.ndarray:
    """fold k 以外のOOFで統合方法を学習して fold k を予測し、アンサンブルのOOFを作る。"""
    oof = np.full(preds.shape[1:], np.nan)
    for k in np.unique(fold_ids[fold_ids >= 0]):
        train = (fold_ids >= 0) & (fold_ids != k)
        valid = fold_ids == k
        blender = blender_factory().fit(preds[:, train], y[train])
        oof[valid] = blender.predict(preds[:, valid])
    return oof


def run_ensemble(
    config: EnsembleConfig,
    members: list[MemberPredictions] | None = None,
    tracker: Tracker | None = None,
    output_root: Path | None = None,
    experiments_root: Path | None = None,
) -> EnsembleResult:
    """アンサンブルを実行し、予測とスコアを保存する。

    Args:
        config: アンサンブル設定。
        members: 構成要素の予測（Noneなら設定に従って読み込む）。
        tracker: 実験ログの記録先。
        output_root: 出力のルート（Noneなら `outputs/ensembles`）。
        experiments_root: `experiment` 指定の構成要素を探すルート
            （Noneなら `outputs/experiments`）。

    Returns:
        アンサンブル結果。
    """
    if members is None:
        members = [
            load_member(m, experiments_root, config.tracking.tracking_uri) for m in config.members
        ]
    check_members_consistent(members)
    tracker = tracker or NullTracker()
    ref = members[0]
    fold_ids, y = ref.fold_ids, ref.target
    predicted = fold_ids >= 0
    n_classes = ref.oof.shape[1] if ref.oof.ndim == 2 else None
    oof_stack = np.stack([m.oof for m in members])
    metrics = [get_metric(name) for name in config.metrics]

    def factory() -> Blender:
        return make_blender(config, n_classes)

    ens_oof = cross_fit_blend(factory, oof_stack, y, fold_ids)
    final = factory().fit(oof_stack[:, predicted], y[predicted])
    test_pred = None
    if ref.test is not None:
        test_pred = final.predict(np.stack([m.test for m in members if m.test is not None]))

    weights: dict[str, float] | None = None
    if isinstance(final, WeightedBlender) and final.weights_ is not None:
        weights = {m.name: float(w) for m, w in zip(members, final.weights_, strict=True)}
    elif isinstance(final, MeanBlender):
        weights = {m.name: 1.0 / len(members) for m in members}

    rows = [
        {"model": m.name, **{mt.name: mt(y[predicted], m.oof[predicted]) for mt in metrics}}
        for m in members
    ]
    rows.append(
        {
            "model": f"ensemble_{config.method}",
            **{mt.name: mt(y[predicted], ens_oof[predicted]) for mt in metrics},
        }
    )
    scores = pl.DataFrame(rows)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (output_root or outputs_dir() / "ensembles") / config.name / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    tags = {"type": "ensemble", "method": config.method, "task": str(config.task)}
    with tracker.start_run(run_name=config.name, tags=tags):
        tracker.log_params(
            {
                "config": config.model_dump(mode="json"),
                "sources": {m.name: m.source for m in members},
            }
        )
        ens_scores = scores.row(-1, named=True)
        tracker.log_metrics({f"oof_{k}": v for k, v in ens_scores.items() if k != "model"})
        if weights is not None:
            tracker.log_metrics({f"weight_{k}": v for k, v in weights.items()})
        paths = _save_ensemble_outputs(
            config, members, ens_oof, test_pred, scores, weights, output_dir
        )
        for path in paths:
            tracker.log_artifact(path)
        run_id = tracker.active_run_id()

    return EnsembleResult(
        oof_pred=ens_oof,
        test_pred=test_pred,
        scores=scores,
        weights=weights,
        output_dir=output_dir,
        run_id=run_id,
    )


def _save_ensemble_outputs(
    config: EnsembleConfig,
    members: list[MemberPredictions],
    ens_oof: np.ndarray,
    test_pred: np.ndarray | None,
    scores: pl.DataFrame,
    weights: dict[str, float] | None,
    output_dir: Path,
) -> list[Path]:
    """アンサンブルの設定・スコア・重み・予測を保存する。"""
    ref = members[0]
    config_path = output_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            config.model_dump(mode="json") | {"sources": {m.name: m.source for m in members}},
            f,
            allow_unicode=True,
            sort_keys=False,
        )
    scores_path = output_dir / "ensemble_scores.csv"
    scores.write_csv(scores_path)
    paths = [config_path, scores_path]
    if weights is not None:
        weights_path = output_dir / "weights.csv"
        pl.DataFrame({"model": list(weights), "weight": list(weights.values())}).write_csv(
            weights_path
        )
        paths.append(weights_path)
    oof_frame = predictions_to_frame(ens_oof, fold_ids=ref.fold_ids, target=ref.target)
    paths.append(save_predictions(oof_frame, output_dir / OOF_FILENAME))
    if test_pred is not None:
        test_frame = predictions_to_frame(test_pred, ids=ref.test_ids)
        paths.append(save_predictions(test_frame, output_dir / TEST_FILENAME))
    return paths
