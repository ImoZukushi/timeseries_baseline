"""実験設定（YAML）のスキーマ定義と読み込み。

YAMLの内容をpydanticモデルで検証し、型の誤り・未知のキー・矛盾した組み合わせ
（例: 時系列タスクに時間順序を無視したCVを指定）を実行前にエラーにする。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modeling.metrics import get_metric
from modeling.tasks import Task, is_classification

# 時間順序を保つCV（時系列タスクではこれ以外を禁止する）
TIME_AWARE_CV_METHODS = frozenset({"time_series", "sliding_window", "time_cutoff"})

_CLASSIFICATION_ONLY_METRICS = frozenset({"auc", "logloss", "accuracy", "f1", "f1_macro"})


def _validate_metrics(metrics: list[str], task: Task) -> None:
    """指標名の存在とタスクとの整合性を確認する。

    Raises:
        ValueError: 未登録の指標、またはタスクに合わない指標がある場合。
    """
    for name in metrics:
        try:
            get_metric(name)
        except KeyError as e:
            # pydanticの検証エラーとして報告されるようValueErrorに変換する
            raise ValueError(str(e)) from e
        if name in _CLASSIFICATION_ONLY_METRICS and not is_classification(task):
            raise ValueError(f"指標 {name} は分類タスク専用です")
        if name not in _CLASSIFICATION_ONLY_METRICS and is_classification(task):
            raise ValueError(f"指標 {name} は回帰タスク専用です")


class _StrictModel(BaseModel):
    """未知のキーを禁止する基底クラス（YAMLのtypoを検出するため）。"""

    model_config = ConfigDict(extra="forbid")


class DataConfig(_StrictModel):
    """入力データの設定。

    Attributes:
        train_path: 学習データのパス（リポジトリルートからの相対パスも可）。CSVまたはParquet。
        test_path: 予測対象データのパス（任意）。
        target: 目的変数の列名。
        id_col: 行IDの列名（予測ファイルに出力する。特徴量には含めない）。
        group_col: グループCV用のグループ列名。
            特徴量に含めるかは `feature_cols` / `drop_cols` 次第。
        time_col: 時刻列名。指定すると学習データをこの列で昇順ソートする（時系列CVの前提）。
        feature_cols: 特徴量として使う列（未指定なら target・id_col・drop_cols 以外の全列）。
        drop_cols: 特徴量から除外する列。
    """

    train_path: Path
    test_path: Path | None = None
    target: str
    id_col: str | None = None
    group_col: str | None = None
    time_col: str | None = None
    feature_cols: list[str] | None = None
    drop_cols: list[str] = Field(default_factory=list)


class FeatureStepConfig(_StrictModel):
    """Pipelineに入れる特徴量エンジニアリングの1ステップ。

    Attributes:
        class_path: transformerクラスのdotted path
            （例: `feature_engineering.numeric.LogTransformer`）。YAMLでは `class` キーで書く。
        params: コンストラクタ引数。
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    class_path: str = Field(alias="class")
    params: dict[str, Any] = Field(default_factory=dict)


class CVConfig(_StrictModel):
    """CV分割の設定。

    Attributes:
        method: 分割方法。
            `kfold` / `stratified` / `group` / `stratified_group` / `time_series` /
            `sliding_window` / `time_cutoff`。
        n_splits: 分割数（`time_cutoff` では `cutoffs` の数から決まるため不使用）。
        shuffle: kfold系でシャッフルするか。
        seed: シャッフルの乱数シード。
        gap: 時系列CVで学習期間と検証期間の間に空ける行数。
        max_train_size: `sliding_window` の学習期間の行数（必須）。
        test_size: 時系列CVの検証期間の行数（未指定ならsklearnの既定）。
        time_column: `time_cutoff` で使う時刻列（未指定なら `data.time_col`）。
        cutoffs: `time_cutoff` の検証期間の開始時刻（ISO形式文字列）のリスト。
        valid_end: `time_cutoff` の最後の検証期間の終了時刻（未指定ならデータの最後まで）。
    """

    method: Literal[
        "kfold",
        "stratified",
        "group",
        "stratified_group",
        "time_series",
        "sliding_window",
        "time_cutoff",
    ] = "kfold"
    n_splits: int = Field(default=5, ge=2)
    shuffle: bool = True
    seed: int = 42
    gap: int = Field(default=0, ge=0)
    max_train_size: int | None = Field(default=None, ge=1)
    test_size: int | None = Field(default=None, ge=1)
    time_column: str | None = None
    cutoffs: list[str] | None = None
    valid_end: str | None = None

    @model_validator(mode="after")
    def _check_method_requirements(self) -> CVConfig:
        if self.method == "sliding_window" and self.max_train_size is None:
            raise ValueError("sliding_window には max_train_size の指定が必要です")
        if self.method == "time_cutoff" and not self.cutoffs:
            raise ValueError("time_cutoff には cutoffs の指定が必要です")
        return self


class ModelConfig(_StrictModel):
    """モデルの設定。

    Attributes:
        name: `modeling.models` に登録されたモデル名（例: `lightgbm`）。
        params: モデルのハイパーパラメータ。
        early_stopping_rounds: 検証foldを使ったearly stoppingのラウンド数（未指定なら無効）。
            検証foldで打ち切り回数を決めるため、OOFスコアはわずかに楽観的になる点に注意。
    """

    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    early_stopping_rounds: int | None = Field(default=None, ge=1)


class TuningConfig(_StrictModel):
    """Optunaによるハイパーパラメータチューニングの設定。

    Attributes:
        enabled: チューニングを行うか（CLIの `--tune` でも有効化できる）。
        n_trials: 試行回数。
        timeout: 打ち切り秒数。
        search_space: 探索空間の上書き。`{パラメータ名: {type, low, high, log, choices}}`。
            未指定ならモデルの既定探索空間を使う。
        pruning: fold単位の途中打ち切り（MedianPruner）を行うか。
    """

    enabled: bool = False
    n_trials: int = Field(default=50, ge=1)
    timeout: float | None = None
    search_space: dict[str, dict[str, Any]] | None = None
    pruning: bool = True


class ExplainConfig(_StrictModel):
    """SHAPによるモデル解釈の設定。

    Attributes:
        enabled: SHAPを計算するか。
        max_samples: SHAP計算に使う最大サンプル数（計算量削減のためサンプリングする）。
    """

    enabled: bool = True
    max_samples: int = Field(default=2000, ge=1)


class TrackingConfig(_StrictModel):
    """実験ログの設定。

    Attributes:
        enabled: MLflowに記録するか。
        experiment_name: MLflowの実験名（未指定なら `ExperimentConfig.name`）。
        tracking_uri: MLflowのtracking URI（未指定ならリポジトリ直下の `mlruns/mlflow.db`）。
    """

    enabled: bool = True
    experiment_name: str | None = None
    tracking_uri: str | None = None


class ExperimentConfig(_StrictModel):
    """1実験の設定全体。

    Attributes:
        name: 実験名（出力ディレクトリ名・MLflowのrun名に使う）。
        task: 予測タスク。
        data: 入力データの設定。
        features: 特徴量エンジニアリングのステップ（この順にPipelineへ入る）。
        cv: CV分割の設定。
        model: モデルの設定。
        metrics: 評価指標名のリスト。先頭がチューニング・アンサンブルで最適化する主指標。
        test_prediction: テスト予測の作り方。`fold_mean`（foldモデルの平均）または
            `refit_full`（全学習データで再学習したモデル1つで予測）。
        seed: モデルの乱数シード（モデルparamsで明示されていればそちらを優先）。
        tuning: チューニング設定。
        explain: SHAP設定。
        tracking: 実験ログ設定。
    """

    name: str
    task: Task
    data: DataConfig
    features: list[FeatureStepConfig] = Field(default_factory=list)
    cv: CVConfig = Field(default_factory=CVConfig)
    model: ModelConfig
    metrics: list[str] = Field(min_length=1)
    test_prediction: Literal["fold_mean", "refit_full"] = "fold_mean"
    seed: int = 42
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    explain: ExplainConfig = Field(default_factory=ExplainConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @model_validator(mode="after")
    def _check_consistency(self) -> ExperimentConfig:
        _validate_metrics(self.metrics, self.task)
        # 時系列タスクで時間順序を無視したCVを使うと未来の情報で学習してしまう
        if self.task is Task.TIME_SERIES and self.cv.method not in TIME_AWARE_CV_METHODS:
            raise ValueError(
                f"time_series タスクでは時間順序を保つCV {sorted(TIME_AWARE_CV_METHODS)} "
                f"のいずれかを指定してください（指定値: {self.cv.method}）"
            )
        if self.cv.method in ("group", "stratified_group") and self.data.group_col is None:
            raise ValueError(f"{self.cv.method} には data.group_col の指定が必要です")
        if self.cv.method in ("stratified", "stratified_group") and not is_classification(
            self.task
        ):
            raise ValueError(f"{self.cv.method} は分類タスク専用です")
        if self.cv.method == "time_cutoff" and (self.cv.time_column or self.data.time_col) is None:
            raise ValueError("time_cutoff には cv.time_column または data.time_col が必要です")
        return self

    @property
    def primary_metric(self) -> str:
        """主指標（`metrics` の先頭）。"""
        return self.metrics[0]


def load_experiment_config(path: Path) -> ExperimentConfig:
    """YAMLファイルから実験設定を読み込んで検証する。

    Args:
        path: YAMLファイルのパス。

    Returns:
        検証済みの実験設定。

    Raises:
        pydantic.ValidationError: 設定内容が不正な場合。
    """
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)


class EnsembleMemberConfig(_StrictModel):
    """アンサンブルの構成要素（1実験分の予測）の指定。

    `experiment` / `path` / `run_id` のいずれか1つを指定する。

    Attributes:
        name: 構成要素の表示名（重み・スコア表の列名に使う）。
        experiment: 実験名。`outputs/experiments/{実験名}/` 配下の最新の実行結果を使う。
        path: 予測ファイル（`oof_predictions.parquet` 等）を含むディレクトリ。
        run_id: MLflowのrun ID（artifactから予測ファイルを取得する）。
    """

    name: str
    experiment: str | None = None
    path: Path | None = None
    run_id: str | None = None

    @model_validator(mode="after")
    def _check_single_source(self) -> EnsembleMemberConfig:
        n_sources = sum(v is not None for v in (self.experiment, self.path, self.run_id))
        if n_sources != 1:
            raise ValueError(
                f"{self.name}: experiment / path / run_id のいずれか1つを指定してください"
            )
        return self


class StackingConfig(_StrictModel):
    """スタッキングのメタモデル設定。

    Attributes:
        model: メタモデル（`modeling.models` の登録名とパラメータ）。既定は線形モデル。
    """

    model: ModelConfig = Field(default_factory=lambda: ModelConfig(name="linear"))


class EnsembleConfig(_StrictModel):
    """アンサンブルの設定。

    Attributes:
        name: アンサンブル名（出力ディレクトリ名・MLflowのrun名）。
        task: 予測タスク（構成要素の実験と同じであること）。
        members: 構成要素（2つ以上）。
        method: 統合方法。
            `mean`（単純平均）/ `rank_mean`（順位平均、二値分類のみ）/
            `weighted`（OOFで重みを最適化）/ `stacking`（メタモデル）。
        metrics: 評価指標。先頭が重み最適化の目的関数になる。
        stacking: `method: stacking` のときのメタモデル設定。
        seed: 乱数シード。
        tracking: 実験ログ設定。
    """

    name: str
    task: Task
    members: list[EnsembleMemberConfig] = Field(min_length=2)
    method: Literal["mean", "rank_mean", "weighted", "stacking"] = "weighted"
    metrics: list[str] = Field(min_length=1)
    stacking: StackingConfig = Field(default_factory=StackingConfig)
    seed: int = 42
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    @model_validator(mode="after")
    def _check_consistency(self) -> EnsembleConfig:
        _validate_metrics(self.metrics, self.task)
        if self.method == "rank_mean" and self.task is not Task.BINARY:
            # 順位は確率・予測値の尺度を持たないため、順位だけで評価できるAUC向けに限定する
            raise ValueError("rank_mean は二値分類専用です")
        names = [m.name for m in self.members]
        if len(set(names)) != len(names):
            raise ValueError(f"members の name が重複しています: {names}")
        return self

    @property
    def primary_metric(self) -> str:
        """主指標（`metrics` の先頭）。"""
        return self.metrics[0]


def load_ensemble_config(path: Path) -> EnsembleConfig:
    """YAMLファイルからアンサンブル設定を読み込んで検証する。

    Raises:
        pydantic.ValidationError: 設定内容が不正な場合。
    """
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return EnsembleConfig.model_validate(raw)
