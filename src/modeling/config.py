"""実験設定（YAML）のスキーマ定義と読み込み。

YAMLの内容をpydanticモデルで検証し、型の誤り・未知のキー・矛盾した組み合わせ
（例: 時系列タスクに時間順序を無視したCVを指定）を実行前にエラーにする。

YAMLの読み込み（`load_config_dict`）では、検証の前に次の2つを展開する。特徴量の組み合わせを
いろいろ試すときに、共通部分を写さず差分だけを書けるようにするため。

- `base: <YAML>`（またはリスト）: 別の設定を土台にして、書いたキーだけを上書きする。
  dict は再帰的にマージし、リスト・値は置き換える。パスは書いたファイルからの相対パス。
- `features` の要素 `{use: <ブロック>}`: 特徴量ブロックのファイル（`steps` のリスト）に展開する。
  拡張子なしの名前は `configs/features/<名前>.yaml`、`.yaml` で終わる値は
  書いたファイルからの相対パス。

展開後は普通の設定になるため、出力の `config.yaml` や MLflow には展開済みの設定が残る。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from modeling.metrics import get_metric
from modeling.tasks import Task, is_classification
from util.paths import get_repo_root

_YAML_SUFFIXES = (".yaml", ".yml")


def default_feature_blocks_dir() -> Path:
    """特徴量ブロックの既定の置き場所（`configs/features/`）。"""
    return get_repo_root() / "configs" / "features"


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """YAMLファイルを読み、最上位が mapping であることを確認して返す（空ファイルは空dict）。"""
    if not path.is_file():
        raise ValueError(f"設定ファイルが見つかりません: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: YAMLの最上位はキーと値の組（mapping）にしてください")
    return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """dict を再帰的にマージする（両方が dict のキーは再帰、リスト・値は override で置き換え）。"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _chain_text(chain: Sequence[Path], last: Path) -> str:
    return " → ".join(p.name for p in (*chain, last))


def _expand_feature_blocks(
    steps: Any, origin: Path, blocks_dir: Path, chain: tuple[Path, ...] = ()
) -> Any:
    """特徴量ステップのリストの `{use: ...}` を、ブロックファイルの `steps` に展開する。

    Args:
        steps: `features` の値（リスト以外はそのまま返し、検証は pydantic に任せる）。
        origin: `steps` を書いたファイル（相対パスの基準）。
        blocks_dir: 名前で指定したブロックの置き場所。
        chain: 展開中のブロックファイル（循環の検出用）。

    Raises:
        ValueError: ブロックが無い・循環している・書式が誤っている場合。
    """
    if not isinstance(steps, list):
        return steps
    expanded: list[Any] = []
    for step in steps:
        if not (isinstance(step, dict) and "use" in step):
            expanded.append(step)
            continue
        if set(step) != {"use"} or not isinstance(step["use"], str):
            raise ValueError(
                f"{origin}: use は `- use: <ブロック名>` の形で、"
                f"他のキーを付けずに書いてください: {step}"
            )
        name = step["use"]
        if name.endswith(_YAML_SUFFIXES):
            block_path = (origin.parent / name).resolve()
        else:
            block_path = (blocks_dir / f"{name}.yaml").resolve()
        if block_path in chain:
            raise ValueError(f"特徴量ブロックが循環しています: {_chain_text(chain, block_path)}")
        if not block_path.is_file():
            raise ValueError(f"{origin}: 特徴量ブロックが見つかりません: {name}（{block_path}）")
        block = _read_yaml_mapping(block_path)
        if not isinstance(block.get("steps"), list):
            raise ValueError(
                f"{block_path}: 特徴量ブロックには steps（ステップのリスト）が必要です"
            )
        expanded += _expand_feature_blocks(
            block["steps"], block_path, blocks_dir, (*chain, block_path)
        )
    return expanded


def _resolve_config(path: Path, blocks_dir: Path, chain: tuple[Path, ...]) -> dict[str, Any]:
    """1ファイル分の設定を、`use` を展開し `base` を重ねて返す。"""
    path = path.resolve()
    if path in chain:
        raise ValueError(f"base が循環しています: {_chain_text(chain, path)}")
    raw = _read_yaml_mapping(path)
    # use はファイルごとに展開する（相対パスは、そのuseを書いたファイルが基準）
    if "features" in raw:
        raw["features"] = _expand_feature_blocks(raw["features"], path, blocks_dir)
    bases = raw.pop("base", None)
    if bases is None:
        return raw
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(b, str) for b in bases):
        raise ValueError(f"{path}: base にはYAMLのパス（またはそのリスト）を書いてください")
    merged: dict[str, Any] = {}
    for base in bases:
        base_path = Path(base) if Path(base).is_absolute() else path.parent / base
        merged = _deep_merge(merged, _resolve_config(base_path, blocks_dir, (*chain, path)))
    return _deep_merge(merged, raw)


def load_config_dict(path: Path, feature_blocks_dir: Path | None = None) -> dict[str, Any]:
    """YAMLを読み、`base` の継承と特徴量ブロック `use` を展開した dict を返す（検証はしない）。

    Args:
        path: 設定ファイルのパス。
        feature_blocks_dir: 名前で指定した特徴量ブロックの置き場所
            （Noneなら `configs/features/`）。

    Returns:
        展開済みの設定（`base` キーを含まない）。

    Raises:
        ValueError: 土台・ブロックのファイルが無い、循環している、書式が誤っている場合。
    """
    blocks_dir = feature_blocks_dir or default_feature_blocks_dir()
    return _resolve_config(Path(path), blocks_dir, ())


# 時間順序を保つCV（時系列タスクではこれ以外を禁止する）
TIME_AWARE_CV_METHODS = frozenset({"time_series", "sliding_window", "time_cutoff"})


def _validate_metrics(metrics: list[str], task: Task) -> None:
    """指標名の存在とタスクとの整合性を確認する。

    Raises:
        ValueError: 未登録の指標、またはタスクに合わない指標がある場合。
    """
    for name in metrics:
        try:
            metric = get_metric(name)
        except KeyError as e:
            # pydanticの検証エラーとして報告されるようValueErrorに変換する
            raise ValueError(str(e)) from e
        if task not in metric.tasks:
            supported = sorted(str(t) for t in metric.tasks)
            raise ValueError(f"指標 {name} はタスク {task} に対応していません（対応: {supported}）")


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
        dependence_top_k: 「特徴量の値 vs SHAP値」の散布図を描く特徴量数（重要度上位）。
        correlation_top_k: SHAP値同士の相関行列に含める特徴量数（重要度上位）。
        scatter_matrix_top_k: SHAP値同士の散布図行列に含める特徴量数（重要度上位）。
    """

    enabled: bool = True
    max_samples: int = Field(default=2000, ge=1)
    dependence_top_k: int = Field(default=6, ge=1)
    correlation_top_k: int = Field(default=15, ge=2)
    scatter_matrix_top_k: int = Field(default=5, ge=2)


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


class ClipConfig(_StrictModel):
    """予測値の範囲制限。

    Attributes:
        min: 下限（Noneなら制限なし）。
        max: 上限（Noneなら制限なし）。
    """

    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check_range(self) -> ClipConfig:
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("clip の min は max 以下にしてください")
        return self


class ForecastConfig(_StrictModel):
    """再帰的多段予測の設定（`task: time_series` のみ）。

    目的変数から作る特徴量（ラグ・移動平均・変化率）を定義する。予測時は、予測値を
    次の時点の目的変数として履歴に追加し、これらの特徴量を再計算しながら1ステップずつ進む。

    Attributes:
        series_col: 系列IDの列（複数系列のパネルデータの場合。単一系列なら省略）。
        lags: 目的変数のラグ（1以上）。`{target}_lag_{k}` 列になる。
        rolling_windows: `{target}_lag_1` の後方移動平均の窓幅
            （t-1 から過去w期の平均。現在値 y_t を含まないためリークしない）。
        rate_of_change: `{target}_lag_1` の前期比（(y_{t-1} - y_{t-2}) / y_{t-2}）を加えるか。
        horizon: バックテストで評価する最大ステップ数（Noneなら検証期間全体）。
        clip: 予測値の範囲制限（誤差の暴走を防ぐ。例: 非負の目的変数なら min=0）。
            `target_transform` なしなら再帰中の各ステップに、ありなら元の尺度に戻した
            最終予測に適用する。
        target_transform: 目的変数の変換（`log` / `diff` / `log_diff` / `seasonal_diff` /
            `log_seasonal_diff`。Noneなら変換しない）。指定するとモデルは変換後の系列を
            学習・再帰予測し（ラグ・移動平均も変換後の系列から作る）、予測値を元の尺度に
            戻してから評価・出力する。GBDTは学習範囲の外へ外挿できないため、トレンドの
            ある系列では差分系の変換が有効。
        seasonal_period: 季節差分の周期（`seasonal_diff` / `log_seasonal_diff` で必須）。
        target_offset: 対数系の変換で `log(y + offset)` にする値（0を含む目的変数では1など）。
    """

    series_col: str | None = None
    lags: list[int] = Field(min_length=1)
    rolling_windows: list[int] = Field(default_factory=list)
    rate_of_change: bool = False
    horizon: int | None = Field(default=None, ge=1)
    clip: ClipConfig = Field(default_factory=ClipConfig)
    target_transform: (
        Literal["log", "diff", "log_diff", "seasonal_diff", "log_seasonal_diff"] | None
    ) = None
    seasonal_period: int | None = Field(default=None, ge=1)
    target_offset: float = 0.0

    @model_validator(mode="after")
    def _check_values(self) -> ForecastConfig:
        if any(lag < 1 for lag in self.lags):
            raise ValueError(
                "forecast.lags は1以上で指定してください（0以下は未来の値の参照になる）"
            )
        if any(w < 1 for w in self.rolling_windows):
            raise ValueError("forecast.rolling_windows は1以上で指定してください")
        is_seasonal = self.target_transform in ("seasonal_diff", "log_seasonal_diff")
        if is_seasonal and self.seasonal_period is None:
            raise ValueError(
                f"target_transform: {self.target_transform} には seasonal_period が必要です"
            )
        if (self.rolling_windows or self.rate_of_change) and 1 not in self.lags:
            raise ValueError("rolling_windows / rate_of_change には lags に 1 を含めてください")
        return self


class LearningCurveConfig(_StrictModel):
    """学習曲線（学習データの量を変えたときのスコア）の設定。

    Attributes:
        enabled: 作成するか（学習をやり直すため時間がかかる。既定は無効）。
        train_sizes: 各foldの学習データのうち使う割合（0〜1）。
    """

    enabled: bool = False
    train_sizes: list[float] = Field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])

    @model_validator(mode="after")
    def _check_sizes(self) -> LearningCurveConfig:
        if not self.train_sizes or any(not 0 < s <= 1 for s in self.train_sizes):
            raise ValueError("learning_curve.train_sizes は 0 < 割合 <= 1 で指定してください")
        return self


class ValidationCurveConfig(_StrictModel):
    """検証曲線（1つのハイパーパラメータを動かしたときのスコア）の設定。

    Attributes:
        param: 動かすパラメータ名（例: `learning_rate`。`model__` で始まらない名前は
            モデルのパラメータとみなして自動で `model__` を付ける）。Noneなら作成しない。
        values: パラメータの値のリスト。
    """

    param: str | None = None
    values: list[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_values(self) -> ValidationCurveConfig:
        if self.param is not None and len(self.values) < 2:
            raise ValueError("validation_curve.values には2つ以上の値を指定してください")
        return self


class EvaluationConfig(_StrictModel):
    """誤差評価の可視化（`modeling.evaluation`）の設定。

    Attributes:
        enabled: 誤差評価の図・表を作成するか。
        max_points: 散布図などで描く最大点数（多い場合は無作為に間引く）。
        training_history: 学習の推移（反復ごとの学習・検証の損失）を記録・描画するか。
        learning_curve: 学習曲線の設定。
        validation_curve: 検証曲線の設定。
        max_series: 残差の図（分布・Q-Q・ACF/PACF・時系列の残差診断）を描く最大系列数
            （複数系列の場合）。検定の表は全系列について作る。
        ljung_box_lags: 時系列の残差診断の Ljung-Box検定のラグ。Noneなら自動
            （`min(10, n//5)` と、`forecast.seasonal_period` が点数の半分未満ならその周期）。
        unit_root_regression: 単位根検定（ADF・KPSS）の確定項。`c`（定数）/ `ct`（定数＋トレンド）。
        test_alpha: 検定の判定に使う有意水準。
    """

    enabled: bool = True
    max_points: int = Field(default=5000, ge=100)
    training_history: bool = True
    learning_curve: LearningCurveConfig = Field(default_factory=LearningCurveConfig)
    validation_curve: ValidationCurveConfig = Field(default_factory=ValidationCurveConfig)
    max_series: int = Field(default=4, ge=1)
    ljung_box_lags: list[Annotated[int, Field(ge=1)]] | None = None
    unit_root_regression: Literal["c", "ct"] = "c"
    test_alpha: float = Field(default=0.05, gt=0, lt=1)


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
        forecast: 再帰的多段予測の設定（指定時は検証・テスト期間を再帰予測する）。
        evaluation: 誤差評価の可視化の設定。
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
    forecast: ForecastConfig | None = None
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _check_consistency(self) -> ExperimentConfig:
        _validate_metrics(self.metrics, self.task)
        if self.forecast is not None:
            if self.task is not Task.TIME_SERIES:
                raise ValueError("forecast は task: time_series でのみ指定できます")
            if self.data.time_col is None:
                raise ValueError("forecast には data.time_col の指定が必要です")
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


def load_experiment_config(path: Path, feature_blocks_dir: Path | None = None) -> ExperimentConfig:
    """YAMLファイルから実験設定を読み込んで検証する（`base`・`use` を展開してから検証）。

    Args:
        path: YAMLファイルのパス。
        feature_blocks_dir: 特徴量ブロックの置き場所（Noneなら `configs/features/`）。

    Returns:
        検証済みの実験設定。

    Raises:
        ValueError: `base`・`use` の展開に失敗した場合。
        pydantic.ValidationError: 設定内容が不正な場合。
    """
    return ExperimentConfig.model_validate(load_config_dict(path, feature_blocks_dir))


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
    """YAMLファイルからアンサンブル設定を読み込んで検証する（`base` を展開してから検証）。

    Raises:
        ValueError: `base` の展開に失敗した場合。
        pydantic.ValidationError: 設定内容が不正な場合。
    """
    return EnsembleConfig.model_validate(load_config_dict(path))
