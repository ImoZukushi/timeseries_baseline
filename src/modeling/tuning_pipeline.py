"""探索 → 学習 → アンサンブルを一括で実行するパイプライン。

パイプラインの設定（`modeling.config.TuningPipelineConfig`、`configs/pipelines/*.yaml`）に並べた
構成モデルごとに次を行い、最後に複数の統合方法でアンサンブルして比較する。

1. Optunaでハイパーパラメータを探索する（`modeling.tuning.tune`。試行回数は study 全体の目標で、
   足りない分だけ実行する。探索の条件が変わると別の study になる）
2. 最良パラメータでCV学習し、OOF予測・テスト予測を保存する（`modeling.experiment.run_experiment`）
3. 全構成モデルの予測を、指定した統合方法ごとにアンサンブルする（`modeling.ensemble.run_ensemble`）

アンサンブルの構成要素には、手順2の **今回の実行結果** をパスで渡す（「最新の実行」を探さない）。
実行前に、全構成モデルが同じデータ・同じCV分割で評価されることを確認する
（アンサンブルの重み・スタッキングはOOF予測を行ごとに組み合わせるため）。

出力（`outputs/pipelines/<name>/<日時>/`）:
    - `experiments/<構成モデル>/<日時>/`: 各構成モデルの実験の出力（`tuning_summary.yaml` など）
    - `ensembles/<name>_<方法>/<日時>/`: 統合方法ごとのアンサンブルの出力
    - `pipeline.yaml`: 実行したパイプラインの設定
    - `members.csv`: 構成モデルごとの探索・CVの結果
    - `best_params.yaml`: 構成モデルごとの学習に使ったパラメータ
    - `summary.csv`: 構成モデル単体と各アンサンブルのOOFスコア（主指標の順位つき）
    - `final.yaml`: 主指標が最良のもの（最終モデル）
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from modeling.config import (
    EnsembleConfig,
    EnsembleMemberConfig,
    ExperimentConfig,
    PipelineMemberConfig,
    TuningPipelineConfig,
    load_experiment_config,
)
from modeling.ensemble import EnsembleResult, MemberPredictions, load_member, run_ensemble
from modeling.experiment import ExperimentResult, load_dataset, run_experiment
from modeling.io import TEST_FILENAME
from modeling.metrics import get_metric
from modeling.tracking import NullTracker, Tracker
from util.paths import outputs_dir

# アンサンブルの前提として、全構成モデルでそろっている必要がある data の項目
_SHARED_DATA_FIELDS = ("train_path", "test_path", "target", "id_col", "group_col", "time_col")


@dataclass(frozen=True)
class PipelineMember:
    """読み込み済みの構成モデル。

    Attributes:
        spec: パイプライン設定での指定（試行回数など）。
        config: 構成モデルの実験設定。
    """

    spec: PipelineMemberConfig
    config: ExperimentConfig


@dataclass
class PipelineResult:
    """パイプラインの実行結果。

    Attributes:
        output_dir: 出力ディレクトリ。
        members: 構成モデル名 → 実験結果。
        ensembles: 統合方法 → アンサンブル結果（構成モデルが1つなら空）。
        summary: 構成モデル単体と各アンサンブルのOOFスコア（`summary.csv` の内容）。
        final: 主指標が最良のもの（`final.yaml` の内容）。
    """

    output_dir: Path
    members: dict[str, ExperimentResult]
    ensembles: dict[str, EnsembleResult] = field(default_factory=dict)
    summary: pl.DataFrame = field(default_factory=pl.DataFrame)
    final: dict[str, Any] = field(default_factory=dict)


def load_members(config: TuningPipelineConfig) -> list[PipelineMember]:
    """構成モデルの実験設定を読み込む（`base`・`use` も展開される）。"""
    return [PipelineMember(m, load_experiment_config(m.config)) for m in config.members]


def _shared_settings(cfg: ExperimentConfig) -> dict[str, Any]:
    """アンサンブルの前提として全構成モデルでそろっている必要がある設定。"""
    data = cfg.data.model_dump(mode="json")
    forecast = cfg.forecast
    return {
        "task": str(cfg.task),
        "data": {k: data[k] for k in _SHARED_DATA_FIELDS},
        "cv": cfg.cv.model_dump(mode="json"),
        "forecast": None
        if forecast is None
        else {"series_col": forecast.series_col, "horizon": forecast.horizon},
        "primary_metric": cfg.primary_metric,
    }


def _ensemble_configs(
    config: TuningPipelineConfig, members: list[PipelineMember], paths: dict[str, Path]
) -> dict[str, EnsembleConfig]:
    """統合方法ごとのアンサンブル設定を作る（構成要素は出力ディレクトリのパスで指定）。"""
    first = members[0].config
    return {
        method: EnsembleConfig(
            name=f"{config.name}_{method}",
            task=first.task,
            members=[
                EnsembleMemberConfig(name=m.config.name, path=paths[m.config.name]) for m in members
            ],
            method=method,
            metrics=config.ensemble.metrics or first.metrics,
            stacking=config.ensemble.stacking,
            seed=config.seed,
            tracking=config.tracking,
        )
        for method in config.ensemble.methods
    }


def validate_members(config: TuningPipelineConfig, members: list[PipelineMember]) -> None:
    """学習を始める前に、構成モデルの設定がアンサンブルできる組み合わせかを確認する。

    Raises:
        ValueError: 構成モデルの名前が重複する、またはデータ・CV分割・タスク・主指標が
            構成モデル間で異なる、またはアンサンブルの設定がタスクに合わない場合。
    """
    names = [m.config.name for m in members]
    duplicated = sorted({n for n in names if names.count(n) > 1})
    if duplicated:
        raise ValueError(f"構成モデルの name が重複しています: {duplicated}")
    reference = _shared_settings(members[0].config)
    for member in members[1:]:
        shared = _shared_settings(member.config)
        diff = sorted(k for k in reference if shared[k] != reference[k])
        if diff:
            raise ValueError(
                f"{member.config.name} の {diff} が {members[0].config.name} と異なります。"
                "アンサンブルする構成モデルは、同じデータ・同じCV分割・同じ主指標にしてください"
            )
    if len(members) >= 2:
        # アンサンブル設定の検証（指標とタスクの整合・rank_mean は二値のみ）を先に行う
        _ensemble_configs(config, members, {n: Path(n) for n in names})


def _scores_row(kind: str, name: str, scores: dict[str, float]) -> dict[str, Any]:
    return {"kind": kind, "name": name, **scores}


def _write_yaml(path: Path, data: Any) -> Path:
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


def _member_row(member: PipelineMember, result: ExperimentResult) -> dict[str, Any]:
    cv, tr = result.cv_result, result.tuning_result
    row: dict[str, Any] = {
        "member": member.config.name,
        "model": member.config.model.name,
        "tuned": tr is not None,
        "study_name": None if tr is None else tr.study_name,
        "n_trials": None if tr is None else tr.n_trials,
        "n_new_trials": None if tr is None else tr.n_new_trials,
        f"tuning_best_{member.config.primary_metric}": None if tr is None else tr.best_value,
    }
    row |= {f"cv_mean_{k}": v for k, v in cv.mean_scores().items()}
    row |= {f"cv_std_{k}": v for k, v in cv.std_scores().items()}
    row |= {f"oof_{k}": v for k, v in cv.oof_scores.items()}
    row["output_dir"] = str(result.output_dir)
    return row


def run_tuning_pipeline(
    config: TuningPipelineConfig,
    *,
    member_tracker: Callable[[ExperimentConfig], Tracker] | None = None,
    ensemble_tracker: Tracker | None = None,
    output_root: Path | None = None,
    optuna_dir: Path | None = None,
    skip_tuning: bool = False,
    log: Callable[[str], None] | None = None,
) -> PipelineResult:
    """探索 → 学習 → アンサンブルを一括で実行し、比較の表を保存する。

    Args:
        config: パイプラインの設定。
        member_tracker: 構成モデルの実験設定から記録先を作る関数（Noneなら記録しない）。
        ensemble_tracker: アンサンブルの記録先（Noneなら記録しない）。
        output_root: 出力のルート（Noneなら `outputs/pipelines`）。
        optuna_dir: Optuna studyの保存先（Noneなら `outputs/optuna`）。
        skip_tuning: Trueなら全構成モデルを探索せずに学習する。
        log: 進捗の表示先（Noneなら表示しない）。

    Returns:
        パイプラインの実行結果。

    Raises:
        ValueError: 構成モデルの設定がアンサンブルできない組み合わせの場合（学習の前に検出）。
    """
    say = log or (lambda _: None)
    members = load_members(config)
    validate_members(config, members)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = (output_root or outputs_dir() / "pipelines") / config.name / stamp
    out.mkdir(parents=True, exist_ok=False)
    tags = {"pipeline": config.name, "pipeline_run": stamp}

    # 1-2. 構成モデルごとに探索 → 最良パラメータでCV学習
    results: dict[str, ExperimentResult] = {}
    for member in members:
        cfg = member.config
        do_tune = member.spec.tune and not skip_tuning
        say(f"=== {cfg.name}（{cfg.model.name}、探索{'あり' if do_tune else 'なし'}）===")
        results[cfg.name] = run_experiment(
            cfg,
            load_dataset(cfg),
            tracker=member_tracker(cfg) if member_tracker else NullTracker(),
            output_root=out / "experiments",
            tune_params=do_tune,
            n_trials=member.spec.n_trials,
            timeout=member.spec.timeout,
            optuna_dir=optuna_dir,
            extra_tags=tags,
        )
        result = results[cfg.name]
        if result.tuning_result is not None:
            tr = result.tuning_result
            say(
                f"  探索: study={tr.study_name} 完了試行={tr.n_trials}"
                f"（今回 {tr.n_new_trials}） 最良={tr.best_value:.6g}"
            )
        scores = result.cv_result.mean_scores()
        say("  CV: " + ", ".join(f"{k}={v:.6g}" for k, v in scores.items()))

    member_rows = [_member_row(m, results[m.config.name]) for m in members]
    pl.DataFrame(member_rows).write_csv(out / "members.csv")
    _write_yaml(out / "best_params.yaml", {name: r.cv_result.params for name, r in results.items()})
    _write_yaml(
        out / "pipeline.yaml",
        config.model_dump(mode="json")
        | {"member_outputs": {name: str(r.output_dir) for name, r in results.items()}},
    )

    # 3. 統合方法ごとにアンサンブル（構成要素は今回の出力を使う）
    ensembles: dict[str, EnsembleResult] = {}
    rows: list[dict[str, Any]] = []
    first = members[0].config
    metric_names = config.ensemble.metrics or first.metrics
    if len(members) >= 2:
        paths = {name: r.output_dir for name, r in results.items()}
        loaded: list[MemberPredictions] = [
            load_member(EnsembleMemberConfig(name=name, path=path)) for name, path in paths.items()
        ]
        for method, ens_config in _ensemble_configs(config, members, paths).items():
            say(f"=== アンサンブル: {method} ===")
            ensembles[method] = run_ensemble(
                ens_config,
                members=loaded,
                tracker=ensemble_tracker or NullTracker(),
                output_root=out / "ensembles",
            )
        # 構成モデル単体のスコアは、アンサンブルと同じ行（OOF予測がある行）で計算した値を使う
        first_scores = next(iter(ensembles.values())).scores
        for row in first_scores.head(len(members)).iter_rows(named=True):
            rows.append(_scores_row("member", row.pop("model"), row))
        for method, ens in ensembles.items():
            row = ens.scores.row(-1, named=True)
            row.pop("model")
            rows.append(_scores_row("ensemble", method, row))
    else:
        result = results[first.name]
        oof = {k: v for k, v in result.cv_result.oof_scores.items() if k in metric_names}
        rows.append(_scores_row("member", first.name, oof))

    primary = get_metric(metric_names[0])
    summary = pl.DataFrame(rows).with_columns(
        pl.col(primary.name)
        .rank("min", descending=primary.direction == "maximize")
        .cast(pl.Int64)
        .alias("rank")
    )
    summary.write_csv(out / "summary.csv")

    best = summary.sort("rank").row(0, named=True)
    final: dict[str, Any] = {
        "kind": best["kind"],
        "name": best["name"],
        "metric": primary.name,
        "score": best[primary.name],
        "scores": {m: best[m] for m in metric_names if m in best},
    }
    if best["kind"] == "ensemble":
        ens = ensembles[best["name"]]
        final["weights"] = ens.weights
        final["output_dir"] = str(ens.output_dir)
    else:
        final["output_dir"] = str(results[best["name"]].output_dir)
    test_path = Path(final["output_dir"]) / TEST_FILENAME
    final["test_predictions"] = str(test_path) if test_path.is_file() else None
    _write_yaml(out / "final.yaml", final)
    say(f"最終モデル: {final['kind']} {final['name']}（{primary.name}={final['score']:.6g}）")
    return PipelineResult(
        output_dir=out, members=results, ensembles=ensembles, summary=summary, final=final
    )
