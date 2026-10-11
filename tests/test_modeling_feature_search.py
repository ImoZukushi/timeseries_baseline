"""modeling.feature_search（特徴量の探索）と scripts/run_feature_search.py のテスト。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest
import run_feature_search
import yaml
from pydantic import ValidationError

from modeling.config import load_feature_search_config
from modeling.feature_search import (
    build_patterns,
    validate_patterns,
)
from modeling.feature_search import (
    run_feature_search as run_search,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def search_dir(synthetic_frame: pl.DataFrame, tmp_path: Path) -> Path:
    """合成データで小さく回せる探索（a・b・cat の3グループ）。"""
    train = tmp_path / "train.parquet"
    synthetic_frame.write_parquet(train)
    root = tmp_path / "configs"
    _write_yaml(
        root / "search" / "experiment.yaml",
        {
            "task": "regression",
            "data": {"train_path": str(train), "target": "y_reg", "time_col": "ts"},
            "cv": {"method": "time_series", "n_splits": 3},
            "model": {"name": "lightgbm", "params": {"n_estimators": 30}},
            "metrics": ["rmse", "mae"],
            "explain": {"enabled": False},
            "evaluation": {"enabled": False},
            "tracking": {"enabled": False},
        },
    )
    _write_yaml(
        root / "cat_block.yaml",
        {
            "steps": [
                {
                    "class": "feature_engineering.categorical.PolarsOrdinalEncoder",
                    "params": {"variables": ["cat"]},
                }
            ]
        },
    )
    _write_yaml(
        root / "search.yaml",
        {
            "name": "fs",
            "experiment": "search/experiment.yaml",
            "groups": {
                "ga": ["a"],
                "gb": {"columns": ["b"], "description": "b の列"},
                "gcat": {"columns": ["cat"], "features": [{"use": "cat_block.yaml"}]},
            },
            "patterns": [
                {"name": "p_a", "groups": ["ga"]},
                {"name": "p_ab", "groups": ["ga", "gb"]},
                {"name": "p_all", "groups": ["*"]},
                {
                    "name": "p_ab_lr",
                    "groups": ["ga", "gb"],
                    "overrides": {"model": {"params": {"learning_rate": 0.2}}},
                },
            ],
            "ablation": {"pattern": "p_all", "keep": ["ga"]},
        },
    )
    return root


# --- 設定と組み立て ----------------------------------------------------------------------------


def test_load_and_build_patterns(search_dir: Path) -> None:
    search = load_feature_search_config(search_dir / "search.yaml")
    # experiment はパスから読み込まれ、グループの省略形・use が展開される
    assert search.experiment["task"] == "regression"
    assert search.groups["ga"].columns == ["a"]
    assert search.groups["gcat"].features[0].class_path.endswith("PolarsOrdinalEncoder")
    assert search.baseline_pattern == "p_a"

    patterns = build_patterns(search)
    names = [p.name for p in patterns]
    # アブレーションは keep のグループ以外を1つずつ外したもの
    assert names == ["p_a", "p_ab", "p_all", "p_ab_lr", "p_all__minus_gb", "p_all__minus_gcat"]
    by_name = {p.name: p for p in patterns}
    all_cfg = by_name["p_all"].config
    assert all_cfg.name == "fs__p_all"
    assert all_cfg.data.feature_cols == ["a", "b", "cat"]
    assert [s.class_path for s in all_cfg.features] == [
        "feature_engineering.categorical.PolarsOrdinalEncoder"
    ]
    # グループのステップは、そのグループを使うパターンにだけ入る
    assert by_name["p_all__minus_gcat"].config.features == []
    assert by_name["p_all__minus_gcat"].ablated_group == "gcat"
    # overrides はモデルのパラメータ単位で上書きされる（n_estimators は共通設定のまま）
    assert by_name["p_ab_lr"].config.model.params == {"n_estimators": 30, "learning_rate": 0.2}
    # MLflow の実験名の既定は探索名
    assert all_cfg.tracking.experiment_name == "fs"
    validate_patterns(patterns, ["a", "b", "cat", "y_reg", "ts"])


def test_build_patterns_only_keeps_baseline_and_ablation_base(search_dir: Path) -> None:
    search = load_feature_search_config(search_dir / "search.yaml")
    names = [p.name for p in build_patterns(search, only=["p_ab"], ablation=False)]
    assert names == ["p_a", "p_ab"]
    names = [p.name for p in build_patterns(search, only=["p_ab"])]
    assert names[:3] == ["p_a", "p_ab", "p_all"]
    with pytest.raises(ValueError, match="未定義のパターン"):
        build_patterns(search, only=["nope"])


def test_validate_patterns_reports_missing_columns(search_dir: Path) -> None:
    search = load_feature_search_config(search_dir / "search.yaml")
    with pytest.raises(ValueError, match="学習データに無い列"):
        validate_patterns(build_patterns(search), ["a", "b"])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"patterns": [{"name": "x", "groups": ["missing"]}]}, "未定義のグループ"),
        (
            {"patterns": [{"name": "x", "groups": ["ga"]}, {"name": "x", "groups": ["gb"]}]},
            "重複",
        ),
        ({"baseline": "nope"}, "baseline"),
        ({"ablation": {"pattern": "p_a", "keep": ["nope"]}}, "ablation.keep"),
        ({"groups": {"ga": {"description": "列なし"}}}, "columns か features"),
    ],
)
def test_invalid_search_config(search_dir: Path, change: dict[str, Any], message: str) -> None:
    raw = yaml.safe_load((search_dir / "search.yaml").read_text(encoding="utf-8"))
    raw |= change
    if "groups" in change:
        raw["patterns"] = [{"name": "p", "groups": ["ga"]}]
        raw["ablation"] = {}
    path = _write_yaml(search_dir / "invalid.yaml", raw)
    with pytest.raises((ValidationError, ValueError), match=message):
        load_feature_search_config(path)


def test_overrides_cannot_change_cv_or_data(search_dir: Path) -> None:
    raw = yaml.safe_load((search_dir / "search.yaml").read_text(encoding="utf-8"))
    raw["patterns"].append({"name": "bad", "groups": ["ga"], "overrides": {"cv": {"n_splits": 5}}})
    raw["patterns"].append(
        {"name": "bad2", "groups": ["ga"], "overrides": {"data": {"target": "a"}}}
    )
    path = _write_yaml(search_dir / "bad.yaml", raw)
    search = load_feature_search_config(path)
    with pytest.raises(ValueError, match="変えられません"):
        build_patterns(search)


def test_bundled_hx_dummy_config_is_valid() -> None:
    search = load_feature_search_config(REPO_ROOT / "configs" / "feature_search" / "hx_dummy.yaml")
    patterns = build_patterns(search)
    assert search.baseline_pattern == "01_raw"
    # 機器・月のグループは外さない
    assert all(p.ablated_group != "unit_calendar" for p in patterns)
    validate_patterns(patterns)


# --- 実行 ----------------------------------------------------------------------------------


def test_run_feature_search_writes_outputs(search_dir: Path, tmp_path: Path) -> None:
    search = load_feature_search_config(search_dir / "search.yaml")
    logs: list[str] = []
    result = run_search(search, output_root=tmp_path / "out", log=logs.append)
    out = result.output_dir
    for name in (
        "search.yaml",
        "summary.csv",
        "fold_scores.csv",
        "ablation.csv",
        "feature_importance.csv",
        "summary.png",
        "fold_scores.png",
        "ablation.png",
    ):
        assert (out / name).is_file(), name

    summary = result.summary
    assert summary["rank"].to_list() == sorted(summary["rank"].to_list())
    base = summary.filter(pl.col("pattern") == "p_a").row(0, named=True)
    assert base["improvement_rmse"] == 0 and base["fold_wins"] == 0
    # 目的変数は 2a + b なので、b を加えると基準より良くなる
    ab = summary.filter(pl.col("pattern") == "p_ab").row(0, named=True)
    assert ab["improvement_rmse"] > 0 and ab["n_columns"] == 2 and ab["n_model_features"] == 2
    # 同じ設定（p_all__minus_gcat と p_ab）は1回だけ学習し、結果を使い回す
    assert result.experiments["p_all__minus_gcat"] is result.experiments["p_ab"]
    assert any("同じ設定" in line for line in logs)
    # b を外すと悪くなる（効いているグループ）
    loss = dict(
        zip(result.ablation["ablated_group"], result.ablation["loss_when_removed"], strict=True)
    )
    assert loss["gb"] > 0
    assert set(result.fold_scores["pattern"]) == set(summary["pattern"])
    importance = result.importance.filter(pl.col("pattern") == "p_ab")
    assert set(importance["feature"]) == {"a", "b"}
    assert importance["importance_mean"].sum() == pytest.approx(1.0)
    # 各パターンの実験の出力は探索の出力の下にまとまる
    assert (out / "experiments" / "fs__p_ab").is_dir()
    saved = yaml.safe_load((out / "search.yaml").read_text(encoding="utf-8"))
    assert [p["name"] for p in saved["resolved_patterns"]] == list(result.experiments)


def test_cli_check_and_run(
    search_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = str(search_dir / "search.yaml")
    run_feature_search.main(["--config", config, "--check"])
    assert "設定の検証のみ" in capsys.readouterr().out
    run_feature_search.main(
        [
            "--config",
            config,
            "--patterns",
            "p_ab",
            "--no-ablation",
            "--no-tracking",
            "--output-root",
            str(tmp_path / "cli"),
        ]
    )
    out = capsys.readouterr().out
    assert "p_ab" in out and "パターンの比較" in out
    assert list((tmp_path / "cli" / "fs").glob("*/summary.csv"))
