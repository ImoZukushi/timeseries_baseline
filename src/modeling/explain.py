"""SHAPによるモデル解釈。

CVの各foldモデルを、そのfoldの**検証データ**に対して説明し、全foldを連結した
「OOF SHAP」を作る。学習に使ったデータで説明すると過学習した関係まで重要に見えるため、
予測性能の評価（OOF）と同じ考え方で解釈にも未学習データを使う。

- 木モデル（`ModelSpec.explainer_kind == "tree"`）: `shap.TreeExplainer`（高速・厳密）
- それ以外: `shap.Explainer` のpermutation法（モデル非依存。予測関数だけを使う）

SHAP値のスケールは、木モデルではモデルの生出力（回帰は予測値、分類はlog-odds）、
permutation法では `modeling.tasks.predict` の出力（分類は確率）になる。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap

from modeling.config import ExperimentConfig
from modeling.cv import Fold
from modeling.models import get_model_spec
from modeling.pipeline import MODEL_STEP
from modeling.tasks import Task, predict
from modeling.trainer import CVResult
from util.plotting import add_caption, ensure_japanese_font

# permutation法で背景分布に使うサンプル数
_BACKGROUND_SIZE = 100


@dataclass
class ShapResult:
    """OOF SHAPの計算結果。

    Attributes:
        values: SHAP値。二値・回帰は (n, n_features)、多クラスは (n, n_features, n_classes)。
        base_values: 期待値。二値・回帰は (n,)、多クラスは (n, n_classes)。
        data: SHAP計算に使った特徴量（前処理後、モデル入力そのもの）。
        rows: `data` の各行が元の学習データの何行目か。
        explainer_kind: 使ったexplainerの種類。
    """

    values: np.ndarray
    base_values: np.ndarray
    data: pd.DataFrame
    rows: np.ndarray
    explainer_kind: str

    @property
    def feature_names(self) -> list[str]:
        """特徴量名。"""
        return [str(c) for c in self.data.columns]

    def importance(self, class_names: list[str] | None = None) -> pl.DataFrame:
        """特徴量ごとの平均|SHAP|（重要度の降順）。

        多クラスでは全クラス平均の `mean_abs_shap` に加えて、クラス別の列も持つ。
        """
        abs_values = np.abs(self.values)
        columns: dict[str, Any] = {"feature": self.feature_names}
        if abs_values.ndim == 3:
            columns["mean_abs_shap"] = abs_values.mean(axis=(0, 2))
            names = class_names or [str(k) for k in range(abs_values.shape[2])]
            for k, name in enumerate(names):
                columns[f"mean_abs_shap_{name}"] = abs_values[:, :, k].mean(axis=0)
        else:
            columns["mean_abs_shap"] = abs_values.mean(axis=0)
        return pl.DataFrame(columns).sort("mean_abs_shap", descending=True)


def sample_oof_rows(fold_ids: np.ndarray, max_samples: int, seed: int) -> np.ndarray:
    """OOF予測された行（fold_id >= 0）から最大 `max_samples` 行を無作為抽出する（昇順で返す）。"""
    candidates = np.flatnonzero(fold_ids >= 0)
    if len(candidates) <= max_samples:
        return candidates
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(candidates, size=max_samples, replace=False))


def _explain_tree(model: Any, X: pd.DataFrame, task: Task) -> tuple[np.ndarray, np.ndarray]:
    """TreeExplainerでSHAP値を計算する。"""
    explanation = shap.TreeExplainer(model)(X)
    values = np.asarray(explanation.values, dtype=np.float64)
    base = np.asarray(explanation.base_values, dtype=np.float64)
    if task is Task.BINARY and values.ndim == 3:
        # モデルによっては二値分類でも2クラス分を返すため陽性クラスだけ取り出す
        values, base = values[:, :, 1], base[:, 1]
    return values, np.broadcast_to(base, values.shape[:1] + values.shape[2:]).copy()


def _explain_permutation(
    model: Any,
    X: pd.DataFrame,
    background: pd.DataFrame,
    task: Task,
    n_classes: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """permutation法（モデル非依存）でSHAP値を計算する。"""
    columns = X.columns
    dtypes = X.dtypes

    def predict_fn(values: np.ndarray) -> np.ndarray:
        # SHAPはnumpy配列を渡してくるため、モデルが学習時と同じ列名・型を受け取れるよう戻す
        frame = pd.DataFrame(values, columns=columns).astype(dtypes)
        return predict(model, frame, task, n_classes)

    masker = shap.maskers.Independent(background, max_samples=_BACKGROUND_SIZE)
    explainer = shap.Explainer(predict_fn, masker, algorithm="permutation", seed=seed)
    explanation = explainer(X, max_evals=max(500, 2 * X.shape[1] + 1), silent=True)
    values = np.asarray(explanation.values, dtype=np.float64)
    base = np.asarray(explanation.base_values, dtype=np.float64)
    return values, base


def compute_oof_shap(
    config: ExperimentConfig,
    X: pl.DataFrame,
    folds: Sequence[Fold],
    cv_result: CVResult,
    n_classes: int | None = None,
    max_samples: int | None = None,
) -> ShapResult:
    """各foldモデルをそのfoldの検証データで説明し、連結したSHAP値を返す。

    Args:
        config: 実験設定。
        X: 学習データの特徴量（`run_cv` に渡したもの）。
        folds: `run_cv` に渡したCV分割。
        cv_result: `run_cv` の結果（foldモデルを含む）。
        n_classes: 多クラス分類のクラス数。
        max_samples: 計算に使う最大行数（Noneなら `config.explain.max_samples`）。

    Returns:
        OOF SHAPの計算結果。
    """
    kind = get_model_spec(config.model.name).explainer_kind
    limit = config.explain.max_samples if max_samples is None else max_samples
    rows = sample_oof_rows(cv_result.fold_ids, limit, config.seed)
    values_list: list[np.ndarray] = []
    base_list: list[np.ndarray] = []
    data_list: list[pd.DataFrame] = []
    row_list: list[np.ndarray] = []
    rng = np.random.default_rng(config.seed)
    for fold, ((train_idx, _), pipeline) in enumerate(zip(folds, cv_result.models, strict=False)):
        fold_rows = rows[cv_result.fold_ids[rows] == fold]
        if len(fold_rows) == 0:
            continue
        preprocess, model = pipeline[:-1], pipeline.named_steps[MODEL_STEP]
        Xt = preprocess.transform(X[fold_rows])
        if kind == "tree":
            values, base = _explain_tree(model, Xt, config.task)
        else:
            # 背景分布はそのfoldの学習データから抽出する（モデルが見た分布を基準にする）
            n_bg = min(len(train_idx), _BACKGROUND_SIZE * 10)
            bg_rows = np.sort(rng.choice(train_idx, size=n_bg, replace=False))
            background = preprocess.transform(X[bg_rows])
            values, base = _explain_permutation(
                model, Xt, background, config.task, n_classes, config.seed
            )
        values_list.append(values)
        base_list.append(base)
        data_list.append(Xt)
        row_list.append(fold_rows)

    return ShapResult(
        values=np.concatenate(values_list),
        base_values=np.concatenate(base_list),
        data=pd.concat(data_list, ignore_index=True),
        rows=np.concatenate(row_list),
        explainer_kind=kind,
    )


def _explanation(result: ShapResult, class_index: int | None = None) -> shap.Explanation:
    """描画用の `shap.Explanation` を作る（多クラスは指定クラス分のみ）。"""
    values = result.values if class_index is None else result.values[:, :, class_index]
    base = result.base_values if class_index is None else result.base_values[:, class_index]
    return shap.Explanation(
        values=values,
        base_values=base,
        data=result.data.to_numpy(dtype=np.float64, na_value=np.nan),
        feature_names=result.feature_names,
    )


def plot_importance_bar(importance: pl.DataFrame, title: str, max_display: int = 20) -> plt.Figure:
    """平均|SHAP|の横棒グラフを描く。"""
    ensure_japanese_font()
    top = importance.head(max_display).reverse()
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * top.height + 1.5)), constrained_layout=True)
    ax.barh(top["feature"].to_list(), top["mean_abs_shap"].to_list())
    ax.set_xlim(left=0)  # 棒グラフは0起点
    ax.set_xlabel("平均 |SHAP値|")
    ax.set_title(title)
    return fig


def plot_beeswarm(result: ShapResult, title: str, class_index: int | None = None) -> plt.Figure:
    """SHAPのbeeswarm図（特徴量の値とSHAP値の関係）を描く。"""
    ensure_japanese_font()
    n_display = min(20, len(result.feature_names))
    fig, ax = plt.subplots(figsize=(9, max(3, 0.4 * n_display + 1.5)), constrained_layout=True)
    shap.plots.beeswarm(
        _explanation(result, class_index), max_display=n_display, ax=ax, show=False, plot_size=None
    )
    ax.set_title(title)
    return fig


def save_shap_outputs(
    result: ShapResult,
    output_dir: Path,
    experiment_name: str,
    class_names: list[str] | None = None,
) -> list[Path]:
    """SHAPの重要度表・図・値を保存し、保存したパスのリストを返す。

    Args:
        result: OOF SHAPの計算結果。
        output_dir: 保存先ディレクトリ。
        experiment_name: 図のタイトルに使う実験名。
        class_names: 多クラス分類のクラスラベル（図のタイトル・列名に使う）。

    Returns:
        保存したファイルのパス。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    caption = (
        f"OOF SHAP（各foldモデルをその検証データで説明） n={len(result.rows):,}, "
        f"explainer={result.explainer_kind}"
    )
    importance = result.importance(class_names)
    paths = [output_dir / "shap_importance.csv"]
    importance.write_csv(paths[0])

    figures: list[tuple[str, plt.Figure]] = [
        (
            "shap_importance_bar.png",
            plot_importance_bar(importance, f"{experiment_name}: SHAP重要度（平均|SHAP値|）"),
        )
    ]
    if result.values.ndim == 3:
        names = class_names or [str(k) for k in range(result.values.shape[2])]
        for k, name in enumerate(names):
            figures.append(
                (
                    f"shap_beeswarm_class_{k}.png",
                    plot_beeswarm(result, f"{experiment_name}: SHAP beeswarm（クラス {name}）", k),
                )
            )
    else:
        figures.append(
            ("shap_beeswarm.png", plot_beeswarm(result, f"{experiment_name}: SHAP beeswarm"))
        )
    for filename, fig in figures:
        add_caption(fig, caption)
        path = output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # 後から任意の図を描けるよう、SHAP値そのものも行番号付きで保存する
    values_2d = result.values.reshape(result.values.shape[0], -1)
    if result.values.ndim == 3:
        value_cols = [
            f"{f}__class_{k}" for f in result.feature_names for k in range(result.values.shape[2])
        ]
    else:
        value_cols = result.feature_names
    values_frame = pl.DataFrame(values_2d, schema=value_cols, orient="row").insert_column(
        0, pl.Series("row", result.rows)
    )
    values_path = output_dir / "shap_values.parquet"
    values_frame.write_parquet(values_path)
    paths.append(values_path)
    return paths
