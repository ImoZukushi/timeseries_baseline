"""単一モデルで特徴量の組み合わせを比べる「特徴量の探索」。

探索の設定（`modeling.config.FeatureSearchConfig`、`configs/feature_search/*.yaml`）では、特徴量を
**グループ**（一緒に足し引きする列と、それに伴う特徴量ステップ）に分けて定義し、グループの組み合わせを
**パターン**として並べる。パターンごとに共通の実験設定から実験設定を組み立てて CV 学習し、
同じ CV 分割で評価したスコアを比べる。

- パターンごとの実験設定: 共通の実験設定 ＋ `overrides` に、`data.feature_cols`（グループの列）と
  `features`（共通の実験設定のステップ → グループのステップ → パターンのステップ）を加えたもの。
- アブレーション: 指定したパターンから、グループを1つずつ外したパターンを自動で加える
  （外すとスコアが悪くなるグループほど、そのパターンの中で効いている）。
- 比較: 基準のパターン（`baseline`）との主指標の差を、fold 平均と fold ごとの勝ち数で示す。
  全パターンが同じデータ・同じ CV 分割を使うことを、学習の前に確認する。

出力（`outputs/feature_search/<name>/<日時>/`）:
    - `experiments/<name>__<パターン>/<日時>/`: 各パターンの実験の出力（`run_experiment` と同じ）
    - `search.yaml`: 探索の設定と、各パターンのグループ・列
    - `summary.csv`: パターンごとのスコア・基準との差・fold の勝ち数（主指標の順位つき）
    - `fold_scores.csv`: パターン × fold のスコア（縦持ち）
    - `ablation.csv`: アブレーションの結果（外したグループ・土台との差）
    - `feature_importance.csv`: パターン × 特徴量の重要度（fold 平均の割合。GBDT のみ）
    - `summary.png` ・ `fold_scores.png` ・ `ablation.png`: 比較の図
"""

from __future__ import annotations

import copy
import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import japanize_matplotlib  # noqa: F401  # 日本語フォント（IPAexGothic）を matplotlib に登録する
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml
from sklearn.pipeline import Pipeline

from modeling.config import (
    ExperimentConfig,
    FeaturePatternConfig,
    FeatureSearchConfig,
    _deep_merge,
)
from modeling.experiment import ExperimentResult, load_table, prepare_dataset, run_experiment
from modeling.metrics import get_metric
from modeling.pipeline import MODEL_STEP
from modeling.tracking import NullTracker, Tracker
from util.paths import ensure_parent_dir, outputs_dir
from util.plotting import add_caption, ensure_japanese_font

# 比較の前提として、全パターンでそろっている必要がある data の項目
_SHARED_DATA_FIELDS = ("train_path", "test_path", "target", "id_col", "group_col", "time_col")


@dataclass(frozen=True)
class SearchPattern:
    """実行する1パターン（実験設定に組み立て済み）。

    Attributes:
        name: パターン名。
        groups: 使う特徴量グループ名。
        config: 実験設定（`name` は `<探索名>__<パターン名>`）。
        ablated_group: アブレーションで外したグループ（通常のパターンは None）。
    """

    name: str
    groups: list[str]
    config: ExperimentConfig
    ablated_group: str | None = None


@dataclass
class FeatureSearchResult:
    """特徴量の探索の結果。

    Attributes:
        output_dir: 出力ディレクトリ。
        experiments: パターン名 → 実験結果。
        summary: パターンごとのスコアと基準との差（`summary.csv` の内容）。
        fold_scores: パターン × fold のスコア（`fold_scores.csv` の内容）。
        ablation: アブレーションの結果（行わなければ空）。
        importance: パターン × 特徴量の重要度（取得できなければ空）。
    """

    output_dir: Path
    experiments: dict[str, ExperimentResult] = field(default_factory=dict)
    summary: pl.DataFrame = field(default_factory=pl.DataFrame)
    fold_scores: pl.DataFrame = field(default_factory=pl.DataFrame)
    ablation: pl.DataFrame = field(default_factory=pl.DataFrame)
    importance: pl.DataFrame = field(default_factory=pl.DataFrame)


# --- パターンの組み立て ----------------------------------------------------------------


def _unique(items: list[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


def build_pattern_config(
    search: FeatureSearchConfig,
    pattern: FeaturePatternConfig,
    groups: list[str] | None = None,
    name: str | None = None,
) -> ExperimentConfig:
    """パターンの実験設定を組み立てる。

    Args:
        search: 探索の設定。
        pattern: パターン（`overrides` ・ `features` を使う）。
        groups: 使うグループ（Noneならパターンの groups。アブレーションで差し替える）。
        name: パターン名（Noneならパターンの name）。

    Returns:
        検証済みの実験設定。

    Raises:
        ValueError: `overrides` でデータ・CV・指標などの比較の前提を変えようとした場合。
        pydantic.ValidationError: 組み立てた実験設定が不正な場合。
    """
    forbidden = {"name", "task", "cv", "metrics", "features"} & set(pattern.overrides)
    data_overrides = set(pattern.overrides.get("data", {}) or {})
    if forbidden or data_overrides & {*_SHARED_DATA_FIELDS, "feature_cols"}:
        raise ValueError(
            f"パターン {pattern.name}: overrides では {sorted(forbidden | data_overrides)} を"
            "変えられません（特徴量は groups / features で指定してください）"
        )
    groups = search.pattern_groups(pattern) if groups is None else groups
    raw = _deep_merge(copy.deepcopy(search.experiment), copy.deepcopy(pattern.overrides))
    columns = _unique([c for g in groups for c in search.groups[g].columns])
    steps = [
        *raw.get("features", []),
        *[s.model_dump(by_alias=True) for g in groups for s in search.groups[g].features],
        *[s.model_dump(by_alias=True) for s in pattern.features],
    ]
    raw["name"] = f"{search.name}__{name or pattern.name}"
    raw["features"] = steps
    raw["data"] = {**raw.get("data", {}), "feature_cols": columns}
    tracking = raw.setdefault("tracking", {})
    tracking.setdefault("experiment_name", search.name)
    return ExperimentConfig.model_validate(raw)


def build_patterns(
    search: FeatureSearchConfig,
    only: list[str] | None = None,
    ablation: bool = True,
) -> list[SearchPattern]:
    """実行するパターン（アブレーションで加えるものを含む）を組み立てる。

    Args:
        search: 探索の設定。
        only: 実行するパターン名に絞る（Noneなら全パターン）。基準のパターンとアブレーションの
            土台は、指定がなくても比較のために含める。
        ablation: アブレーションのパターンを加えるか。

    Returns:
        パターンのリスト（設定の順。アブレーションのパターンは末尾）。

    Raises:
        ValueError: `only` に未定義のパターンがある場合、または組み立てに失敗した場合。
    """
    by_name = {p.name: p for p in search.patterns}
    selected = list(by_name)
    if only is not None:
        unknown = [n for n in only if n not in by_name]
        if unknown:
            raise ValueError(f"未定義のパターン: {unknown}（定義済み: {list(by_name)}）")
        required = {search.baseline_pattern, *only}
        if ablation and search.ablation.pattern is not None:
            required.add(search.ablation.pattern)
        selected = [n for n in by_name if n in required]
    patterns = [
        SearchPattern(
            n, search.pattern_groups(by_name[n]), build_pattern_config(search, by_name[n])
        )
        for n in selected
    ]
    if ablation and search.ablation.pattern is not None:
        base = by_name[search.ablation.pattern]
        base_groups = search.pattern_groups(base)
        for group in base_groups:
            if group in search.ablation.keep:
                continue
            groups = [g for g in base_groups if g != group]
            name = f"{base.name}__minus_{group}"
            config = build_pattern_config(search, base, groups=groups, name=name)
            patterns.append(SearchPattern(name, groups, config, ablated_group=group))
    return patterns


def _shared_settings(cfg: ExperimentConfig) -> dict[str, Any]:
    data = cfg.data.model_dump(mode="json")
    return {
        "task": str(cfg.task),
        "data": {k: data[k] for k in _SHARED_DATA_FIELDS},
        "cv": cfg.cv.model_dump(mode="json"),
        "metrics": cfg.metrics,
        "forecast": None if cfg.forecast is None else cfg.forecast.model_dump(mode="json"),
    }


def validate_patterns(patterns: list[SearchPattern], columns: list[str] | None = None) -> None:
    """全パターンが同じ条件で比べられるか、使う列がデータにあるかを確認する。

    Args:
        patterns: `build_patterns` の出力。
        columns: 学習データの列（Noneなら列の確認はしない）。

    Raises:
        ValueError: データ・CV・指標がパターン間で異なる、または列がデータに無い場合。
    """
    reference = _shared_settings(patterns[0].config)
    for p in patterns[1:]:
        shared = _shared_settings(p.config)
        diff = sorted(k for k in reference if shared[k] != reference[k])
        if diff:
            raise ValueError(f"パターン {p.name} の {diff} が {patterns[0].name} と異なります")
    if columns is not None:
        available = set(columns)
        for p in patterns:
            missing = [c for c in p.config.data.feature_cols or [] if c not in available]
            if missing:
                raise ValueError(f"パターン {p.name}: 学習データに無い列 {missing}")


# --- 結果の集計 ----------------------------------------------------------------------


def _better(metric: str) -> float:
    """「良くなった」方向の符号（大きいほど良い指標なら +1、小さいほど良いなら −1）。"""
    return 1.0 if get_metric(metric).greater_is_better else -1.0


def fold_score_table(results: dict[str, ExperimentResult]) -> pl.DataFrame:
    """パターン × fold × 指標のスコア（縦持ち）。"""
    rows = [
        {"pattern": name, "fold": i, "metric": metric, "score": float(score)}
        for name, result in results.items()
        for metric, scores in result.cv_result.fold_scores.items()
        for i, score in enumerate(scores)
    ]
    return pl.DataFrame(rows)


def summarize(
    patterns: list[SearchPattern],
    results: dict[str, ExperimentResult],
    baseline: str,
) -> pl.DataFrame:
    """パターンごとのスコアと、基準のパターンとの差をまとめる。

    `improvement_<主指標>` は「基準より良くなった量」（小さいほど良い指標では符号を反転する）。
    `fold_wins` は基準より良かった fold の数。

    Args:
        patterns: 実行したパターン。
        results: パターン名 → 実験結果。
        baseline: 基準のパターン名。

    Returns:
        1行 = 1パターン（主指標の fold 平均が良い順。`rank` 列つき）。
    """
    first = patterns[0].config
    primary = first.primary_metric
    sign = _better(primary)
    base_scores = np.asarray(results[baseline].cv_result.fold_scores[primary])
    rows = []
    for p in patterns:
        cv = results[p.name].cv_result
        scores = np.asarray(cv.fold_scores[primary])
        diff = (scores - base_scores) * sign
        row: dict[str, Any] = {
            "pattern": p.name,
            "groups": ",".join(p.groups),
            "ablated_group": p.ablated_group,
            "n_columns": len(p.config.data.feature_cols or []),
            "n_model_features": _n_model_features(cv.models[0]) if cv.models else None,
        }
        row |= {f"cv_mean_{k}": v for k, v in cv.mean_scores().items()}
        row |= {f"cv_std_{k}": v for k, v in cv.std_scores().items()}
        row |= {f"oof_{k}": v for k, v in cv.oof_scores.items()}
        row[f"improvement_{primary}"] = float(diff.mean())
        row["fold_wins"] = int((diff > 0).sum())
        row["n_folds"] = len(scores)
        row["output_dir"] = str(results[p.name].output_dir)
        rows.append(row)
    summary = pl.DataFrame(rows)
    return summary.with_columns(
        pl.col(f"cv_mean_{primary}")
        .rank(method="min", descending=sign > 0)
        .cast(pl.Int32)
        .alias("rank")
    ).sort("rank")


def ablation_table(summary: pl.DataFrame, base_pattern: str, primary: str) -> pl.DataFrame:
    """アブレーションの結果（グループを外したときに、土台よりどれだけ悪くなったか）。

    `loss_when_removed` が正なら、そのグループを外すと悪くなる（＝そのグループは効いている）。

    Args:
        summary: `summarize` の出力。
        base_pattern: アブレーションの土台のパターン名。
        primary: 主指標。

    Returns:
        1行 = 外したグループ（効いている順）。
    """
    col = f"improvement_{primary}"
    base_improvement = summary.filter(pl.col("pattern") == base_pattern)[col].item()
    return (
        summary.filter(pl.col("ablated_group").is_not_null())
        .select(
            "ablated_group",
            "pattern",
            f"cv_mean_{primary}",
            (base_improvement - pl.col(col)).alias("loss_when_removed"),
        )
        .sort("loss_when_removed", descending=True)
    )


def _n_model_features(model: Pipeline) -> int | None:
    names = model_feature_names(model)
    return None if names is None else len(names)


def model_feature_names(model: Pipeline) -> list[str] | None:
    """学習済み Pipeline がモデルに渡した特徴量の名前（取得できなければ None）。"""
    step = model.named_steps.get("to_model_input")
    names = getattr(step, "feature_names_in_", None)
    return None if names is None else [str(n) for n in names]


def model_importance(model: Pipeline) -> np.ndarray | None:
    """学習済み Pipeline のモデルの特徴量重要度（合計 1 に正規化。取得できなければ None）。

    LightGBM は分割による損失の減少量（gain）、それ以外は `feature_importances_` を使う。
    """
    estimator = model.named_steps.get(MODEL_STEP)
    booster = getattr(estimator, "booster_", None)
    if booster is not None:
        values = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    else:
        raw = getattr(estimator, "feature_importances_", None)
        if raw is None:
            return None
        values = np.asarray(raw, dtype=float)
    total = values.sum()
    return values / total if total > 0 else values


def importance_table(results: dict[str, ExperimentResult]) -> pl.DataFrame:
    """パターン × 特徴量の重要度（fold モデルの平均と標準偏差、パターン内の順位）。"""
    frames = []
    for name, result in results.items():
        cv = result.cv_result
        n_folds = len(next(iter(cv.fold_scores.values())))
        rows = []
        for fold, model in enumerate(cv.models[:n_folds]):
            names, values = model_feature_names(model), model_importance(model)
            if names is None or values is None or len(names) != len(values):
                continue
            rows += [
                {"feature": f, "fold": fold, "importance": float(v)}
                for f, v in zip(names, values, strict=True)
            ]
        if not rows:
            continue
        frames.append(
            pl.DataFrame(rows)
            .group_by("feature")
            .agg(
                pl.col("importance").mean().alias("importance_mean"),
                pl.col("importance").std().alias("importance_std"),
            )
            .with_columns(
                pl.lit(name).alias("pattern"),
                pl.col("importance_mean")
                .rank(method="min", descending=True)
                .cast(pl.Int32)
                .alias("rank"),
            )
            .sort("rank")
        )
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames).select(
        "pattern", "rank", "feature", "importance_mean", "importance_std"
    )


# --- 図 ------------------------------------------------------------------------------


def plot_summary(summary: pl.DataFrame, primary: str, baseline: str) -> plt.Figure:
    """パターンごとの主指標（fold 平均 ± 標準偏差）と、基準との差（fold の勝ち数つき）。"""
    ensure_japanese_font()
    data = summary.sort("rank", descending=True)  # 横棒グラフは下から並ぶため、良い順を上にする
    n = data.height
    fig, axes = plt.subplots(
        1, 2, figsize=(14, max(3.0, 0.45 * n + 1.5)), sharey=True, constrained_layout=True
    )
    y = np.arange(n)
    colors = ["#c44e52" if p == baseline else "#4c72b0" for p in data["pattern"]]
    axes[0].barh(y, data[f"cv_mean_{primary}"], xerr=data[f"cv_std_{primary}"], color=colors)
    axes[0].set_yticks(y, data["pattern"].to_list())
    axes[0].set_xlabel(f"{primary}（fold 平均 ± 標準偏差）")
    axes[0].set_title(f"{primary} のスコア")
    improvement = data[f"improvement_{primary}"]
    axes[1].barh(y, improvement, color=["#55a868" if v > 0 else "#c44e52" for v in improvement])
    axes[1].axvline(0, color="gray", lw=0.8)
    for yi, (v, wins, total) in enumerate(
        zip(improvement, data["fold_wins"], data["n_folds"], strict=True)
    ):
        axes[1].annotate(
            f" {wins}/{total}",
            (v, yi),
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=9,
        )
    # 棒の先の注記（勝ち数）が枠に収まるよう、横軸に余白を取る
    values = improvement.to_numpy()
    low, high = min(0.0, float(values.min())), max(0.0, float(values.max()))
    pad = (high - low) * 0.12 or 1.0
    axes[1].set_xlim(low - (pad if low < 0 else 0), high + pad)
    axes[1].set_xlabel(f"基準（{baseline}）より良くなった量")
    axes[1].set_title("基準との差（数字は基準に勝った fold の数）")
    fig.suptitle("特徴量のパターンの比較")
    add_caption(
        fig, "赤: 基準のパターン。全パターン同じ CV 分割。差は fold ごとの差の平均（正が良い）"
    )
    return fig


def plot_fold_scores(
    fold_scores: pl.DataFrame,
    primary: str,
    baseline: str,
    ablated: set[str] | None = None,
) -> plt.Figure:
    """fold ごとの主指標（パターンごとの折れ線。fold の難しさの違いと順位の入れ替わりを見る）。

    Args:
        fold_scores: `fold_score_table` の出力。
        primary: 主指標。
        baseline: 基準のパターン名（太い黒線で描く）。
        ablated: アブレーションのパターン名（破線で描く）。
    """
    ensure_japanese_font()
    ablated = ablated or set()
    data = fold_scores.filter(pl.col("metric") == primary)
    names = data["pattern"].unique(maintain_order=True).to_list()
    # パターンが多くても見分けやすいよう 20 色のカラーマップを使う（基準は黒）
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    for i, name in enumerate(names):
        part = data.filter(pl.col("pattern") == name).sort("fold")
        if name == baseline:
            style: dict[str, Any] = {"lw": 2.5, "color": "black"}
        else:
            style = {
                "lw": 1.0,
                "alpha": 0.85,
                "color": cmap(i % cmap.N),
                "ls": "--" if name in ablated else "-",
            }
        ax.plot(part["fold"], part["score"], marker="o", label=name, **style)
    ax.set_xticks(sorted(data["fold"].unique().to_list()))
    ax.set_xlabel("fold")
    ax.set_ylabel(primary)
    ax.set_title(f"fold ごとの {primary}（太い黒線: 基準 {baseline}、破線: アブレーション）")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    return fig


def plot_ablation(ablation: pl.DataFrame, base_pattern: str) -> plt.Figure:
    """アブレーション: グループを外したときに悪くなった量（大きいほど効いているグループ）。"""
    ensure_japanese_font()
    data = ablation.sort("loss_when_removed")
    fig, ax = plt.subplots(
        figsize=(10, max(3.0, 0.45 * data.height + 1.5)), constrained_layout=True
    )
    values = data["loss_when_removed"]
    ax.barh(
        data["ablated_group"].to_list(),
        values,
        color=["#55a868" if v > 0 else "#c44e52" for v in values],
    )
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("外したときに悪くなった量（正: 効いている、負: 外した方が良い）")
    ax.set_title(f"アブレーション（土台: {base_pattern}）")
    return fig


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    fig.savefig(ensure_parent_dir(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --- 実行 ----------------------------------------------------------------------------


def _config_key(config: ExperimentConfig) -> str:
    """名前・実験ログの設定を除いた実験設定（学習結果が同じになるかの判定に使う）。"""
    dumped = config.model_dump(mode="json", by_alias=True, exclude={"name", "tracking"})
    # 列の順序も含めて比べる（LightGBM は列の順序で結果がわずかに変わりうるため）
    return yaml.safe_dump(dumped, allow_unicode=True, sort_keys=True)


def _write_yaml(path: Path, data: Any) -> Path:
    with ensure_parent_dir(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


def run_feature_search(
    search: FeatureSearchConfig,
    *,
    only: list[str] | None = None,
    ablation: bool = True,
    tracker_factory: Callable[[ExperimentConfig], Tracker] | None = None,
    output_root: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> FeatureSearchResult:
    """特徴量の探索を実行し、結果を保存する。

    学習データ（・テストデータ）は1回だけ読み込み、パターンごとに使う列を選んで学習する。

    Args:
        search: 探索の設定。
        only: 実行するパターン名に絞る（基準・アブレーションの土台は自動で含める）。
        ablation: アブレーションを行うか（設定の `ablation.pattern` があるとき）。
        tracker_factory: 実験設定 → 実験ログの記録先（Noneなら記録しない）。
        output_root: 出力のルート（Noneなら `outputs/feature_search`）。
        log: 進捗の表示先（Noneなら表示しない）。

    Returns:
        探索の結果。

    Raises:
        ValueError: パターンの組み立て・比較の前提の確認に失敗した場合。
    """
    log = log or (lambda _: None)
    patterns = build_patterns(search, only=only, ablation=ablation)
    first = patterns[0].config
    train = load_table(first.data.train_path)
    test = None if first.data.test_path is None else load_table(first.data.test_path)
    validate_patterns(patterns, train.columns)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = (output_root or outputs_dir() / "feature_search") / search.name / stamp
    out.mkdir(parents=True, exist_ok=False)
    _write_yaml(
        out / "search.yaml",
        search.model_dump(mode="json", by_alias=True)
        | {
            "resolved_patterns": [
                {
                    "name": p.name,
                    "groups": p.groups,
                    "ablated_group": p.ablated_group,
                    "feature_cols": p.config.data.feature_cols,
                }
                for p in patterns
            ]
        },
    )

    results: dict[str, ExperimentResult] = {}
    # 設定（名前以外）が同じパターンは学習結果が同じになるため、1回だけ学習して結果を使い回す
    # （例: アブレーションで外したグループが、別のパターンとちょうど同じ組み合わせになる場合）
    trained: dict[str, str] = {}
    primary = first.primary_metric
    for i, p in enumerate(patterns, start=1):
        log(f"[{i}/{len(patterns)}] {p.name}（列 {len(p.config.data.feature_cols or [])}）")
        key = _config_key(p.config)
        if key in trained:
            results[p.name] = results[trained[key]]
            log(f"    {trained[key]} と同じ設定のため、その結果を使う")
            continue
        trained[key] = p.name
        dataset = prepare_dataset(p.config, train, test)
        tracker = tracker_factory(p.config) if tracker_factory is not None else NullTracker()
        result = run_experiment(
            p.config,
            dataset=dataset,
            tracker=tracker,
            output_root=out / "experiments",
            extra_tags={"feature_search": search.name, "feature_pattern": p.name},
        )
        results[p.name] = result
        cv = result.cv_result
        log(
            f"    {primary}: cv_mean={cv.mean_scores()[primary]:.6f} "
            f"(std={cv.std_scores()[primary]:.6f})"
        )

    baseline = search.baseline_pattern
    summary = summarize(patterns, results, baseline)
    fold_scores = fold_score_table(results)
    importance = importance_table(results)
    summary.write_csv(out / "summary.csv")
    fold_scores.write_csv(out / "fold_scores.csv")
    if not importance.is_empty():
        importance.write_csv(out / "feature_importance.csv")
    _save_figure(plot_summary(summary, primary, baseline), out / "summary.png")
    ablated = {p.name for p in patterns if p.ablated_group is not None}
    _save_figure(plot_fold_scores(fold_scores, primary, baseline, ablated), out / "fold_scores.png")

    ablation_result = pl.DataFrame()
    if any(p.ablated_group is not None for p in patterns):
        base_pattern = search.ablation.pattern
        assert base_pattern is not None  # アブレーションのパターンがあるなら土台も指定されている
        ablation_result = ablation_table(summary, base_pattern, primary)
        ablation_result.write_csv(out / "ablation.csv")
        _save_figure(plot_ablation(ablation_result, base_pattern), out / "ablation.png")

    return FeatureSearchResult(
        output_dir=out,
        experiments=results,
        summary=summary,
        fold_scores=fold_scores,
        ablation=ablation_result,
        importance=importance,
    )
