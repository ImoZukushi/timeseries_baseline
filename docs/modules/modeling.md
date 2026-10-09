# `modeling` — 実験の実行基盤

YAML（または dict）の設定1つから、次の処理をまとめて実行する基盤です。

- CV学習
- 予測の保存
- ハイパーパラメータ探索（Optuna）
- SHAP による解釈
- 誤差評価
- 実験ログ（MLflow）

複数の実験の予測をアンサンブルすることもできます。
モデルはすべて sklearn 互換の estimator として扱い、モデルごとの違いは `ModelSpec` に閉じ込めています。

## 全体の流れ

```
YAML ──load_experiment_config──▶ ExperimentConfig（pydantic で検証）
                                       │
学習・テストの DataFrame ──prepare_dataset──▶ Dataset（X, y, CV分割, …）
                                       │
                               run_experiment
   ├─ tune（tuning 有効時）   Optuna で主指標の fold 平均を最適化
   ├─ cross_validate          fold ごとに Pipeline を学習 → OOF予測・テスト予測
   │    （forecast 指定時は recursive_backtest：検証期間を再帰予測）
   ├─ 予測・スコアの保存       outputs/experiments/{name}/{日時}/
   ├─ save_evaluation_outputs  誤差評価の図・表（evaluation/）
   ├─ compute_oof_shap         OOF SHAP の図・表（shap/）
   └─ Tracker                  MLflow に設定・スコア・ファイルを記録
```

| モジュール | 内容 |
|---|---|
| `config` | 設定のスキーマ（`ExperimentConfig` / `EnsembleConfig`）と YAML の読み込み |
| `tasks` | タスクの種類（`Task`）・目的変数のエンコード・予測値の取り出し |
| `metrics` | 評価指標のレジストリ |
| `cv` | CV分割（`make_folds`・日時カットオフの `TimeCutoffSplit`） |
| `models` | モデル定義 `ModelSpec` とレジストリ（LightGBM・XGBoost・線形・定数・MLP・1次元CNN） |
| `pipeline` | 特徴量ステップ＋モデルの sklearn Pipeline の組み立て |
| `dataset` | 学習用データ一式 `Dataset` と `cross_validate` |
| `trainer` | CV学習ループ（`run_cv`）と結果 `CVResult` |
| `forecasting` | 再帰的多段予測・再帰バックテスト・目的変数の変換 |
| `tuning` | Optuna によるチューニング |
| `explain` | OOF SHAP |
| `evaluation` | CV結果から誤差評価の図・表を作る（部品は `src/evaluation`） |
| `experiment` | 1実験の実行（`prepare_dataset` / `run_experiment`） |
| `ensemble` | 保存済みの予測のアンサンブル |
| `io` | 予測ファイル（Parquet）の保存・読み込み |
| `tracking` | 実験ログ（`MLflowTracker` / `NullTracker`） |

## 使い方

### CLI

```bash
# 実験（CV学習・予測・SHAP・誤差評価・MLflow記録）。--config は複数・globパターン可
uv run python scripts/run_experiment.py --config "configs/experiments/*.yaml"
# チューニング付き（設定の tuning.enabled: true でも可）
uv run python scripts/run_experiment.py --config configs/experiments/example_lgbm.yaml --tune --n-trials 50
# その他: --no-tracking（MLflowに記録しない）, --no-explain（SHAPなし）, --output-root, --optuna-dir

# アンサンブル（構成要素の実験を実行した後に）
uv run python scripts/run_ensemble.py --config configs/ensembles/example_blend.yaml

# 実験ログの閲覧
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

`run_experiment.py` は、実行前にすべての設定を検証します。設定の誤りで長い実行が途中で失敗することはありません。

### Python（スクリプト・ノートブック）

設定は dict からも作れます。読み込み済みの DataFrame を `prepare_dataset` に渡すと、ファイルを経由せずに実行できます。

```python
import numpy as np
import polars as pl

from modeling.config import ExperimentConfig
from modeling.experiment import prepare_dataset, run_experiment
from modeling.tracking import NullTracker
from util.paths import outputs_dir

rng = np.random.default_rng(0)
n = 300
train = pl.DataFrame(
    {
        "id": np.arange(n),
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "cat": rng.choice(["a", "b", "c"], n),
    }
).with_columns((2 * pl.col("x1") - pl.col("x2") + rng.normal(scale=0.3, size=n)).alias("y"))

config = ExperimentConfig.model_validate(
    {
        "name": "doc_example_lgbm",
        "task": "regression",
        "data": {"train_path": "unused.csv", "target": "y", "id_col": "id"},
        "features": [
            {"class": "feature_engineering.categorical.PolarsOrdinalEncoder", "params": {"variables": ["cat"]}}
        ],
        "cv": {"method": "kfold", "n_splits": 3},
        "model": {"name": "lightgbm", "params": {"n_estimators": 200}, "early_stopping_rounds": 20},
        "metrics": ["rmse", "mae"],
        "explain": {"max_samples": 200},
    }
)
dataset = prepare_dataset(config, train)
result = run_experiment(config, dataset, tracker=NullTracker(), output_root=outputs_dir() / "experiments")
print(result.cv_result.oof_scores)  # {'rmse': ..., 'mae': ...}
print(result.output_dir)            # outputs/experiments/doc_example_lgbm/{日時}/
```

YAML のファイルから読むときは、`load_experiment_config(path)` と `load_dataset(config)` を使います（`data.train_path` / `test_path` を読む）。
MLflow に記録するときは、`tracker=MLflowTracker(experiment_name)` を渡します。

## 設定リファレンス（`ExperimentConfig`）

未知のキーはエラーになります（YAML の書き間違いを検出するため）。
矛盾した組み合わせも実行前にエラーにします。例: 時系列タスクに `kfold`、`group` CV に `group_col` が無い、など。

### トップレベル

| キー | 既定 | 内容 |
|---|---|---|
| `name` | 必須 | 実験名（出力ディレクトリ名・MLflow の run 名） |
| `task` | 必須 | `binary` / `multiclass` / `regression` / `time_series` |
| `data` | 必須 | 入力データ（下表） |
| `features` | `[]` | 特徴量のステップ（`{class: dotted path, params: {...}}` のリスト。この順に Pipeline に入る） |
| `cv` | `kfold` 5分割 | CV分割（下表） |
| `model` | 必須 | `{name, params, early_stopping_rounds}` |
| `metrics` | 必須 | 評価指標名のリスト。**先頭が主指標**（チューニング・アンサンブルで最適化） |
| `test_prediction` | `fold_mean` | テスト予測の作り方。`fold_mean`（fold モデルの平均）/ `refit_full`（全学習データで再学習。木の本数は fold の最良反復の平均） |
| `seed` | 42 | モデルの乱数シード（`model.params` で明示した値が優先） |
| `tuning` / `explain` / `tracking` / `forecast` / `evaluation` | 下表 | 各機能の設定 |

### `data`

| キー | 内容 |
|---|---|
| `train_path` / `test_path` | 学習・予測対象のファイル（CSV・Parquet。リポジトリルートからの相対パス可） |
| `target` | 目的変数の列 |
| `id_col` | 行ID（予測ファイルに出力する。特徴量には入らない） |
| `group_col` | グループCV用の列 |
| `time_col` | 時刻列。指定するとデータをこの列で昇順に並べ替える（時系列CVの前提） |
| `feature_cols` | 特徴量にする列（未指定なら `target`・`id_col`・`drop_cols` 以外の全列） |
| `drop_cols` | 特徴量から除く列 |

文字列・日時の列は、`features` でエンコードするか削除してください。そうしないとモデルの入力にできず、エラーになります。日時の列は `modeling.pipeline.DropColumns` で削除できます。

### `cv`

| `method` | 分割 | 必要な設定 |
|---|---|---|
| `kfold` | KFold | `n_splits`, `shuffle`, `seed` |
| `stratified` | StratifiedKFold（分類のみ） | 同上 |
| `group` | GroupKFold | `data.group_col` |
| `stratified_group` | StratifiedGroupKFold（分類のみ） | `data.group_col` |
| `time_series` | TimeSeriesSplit（拡大窓） | `n_splits`, `gap`, `test_size` |
| `sliding_window` | TimeSeriesSplit（学習期間を固定長） | `max_train_size`（必須）, `gap`, `test_size` |
| `time_cutoff` | 日時のカットオフで分割（`TimeCutoffSplit`）。各 fold は「カットオフより前で学習、次のカットオフまでを検証」 | `cutoffs`（ISO 形式の日時のリスト）, `valid_end`, `time_column`（既定は `data.time_col`） |

`task: time_series` では、時間順を保つ `time_series` / `sliding_window` / `time_cutoff` だけが使えます。
複数の系列が1つのファイルに混在するデータでは、行番号ではなく日時で区切る `time_cutoff` を使ってください。

CV分割は `prepare_dataset` で一度だけ計算し、チューニングの全試行とアンサンブルの全構成要素で使い回します。

### `model`

| キー | 内容 |
|---|---|
| `name` | 登録済みのモデル名（下の「モデル」） |
| `params` | ハイパーパラメータ（モデルのコンストラクタ引数。NN は skorch の引数） |
| `early_stopping_rounds` | 検証 fold を使った early stopping（NN は学習 fold から切り出した10%を使う） |

early stopping は検証 fold で打ち切り回数を決めるため、OOF スコアはわずかに楽観的になります。

### `tuning`

| キー | 既定 | 内容 |
|---|---|---|
| `enabled` | false | チューニングするか（CLI の `--tune` でも可） |
| `n_trials` / `timeout` | 50 / なし | 試行回数・打ち切り秒数 |
| `search_space` | なし（モデルの既定） | 探索空間の上書き（下の例） |
| `pruning` | true | fold ごとのスコアで見込みの薄い試行を打ち切る（MedianPruner） |

```yaml
tuning:
  enabled: true
  n_trials: 100
  search_space:
    learning_rate: {type: float, low: 0.005, high: 0.2, log: true}
    num_leaves: {type: int, low: 16, high: 128}
    boosting_type: {type: categorical, choices: [gbdt, dart]}
    min_child_samples: {type: fixed, value: 20}
```

study は `outputs/optuna/{name}.db`（SQLite）に保存されます。同じ実験名で再実行すると、続きから探索します。
最良のパラメータで CV をやり直した結果が、実験の結果になります。試行の一覧は `tuning_trials.csv` に保存されます。

### `explain`

| キー | 既定 | 内容 |
|---|---|---|
| `enabled` | true | OOF SHAP を計算するか（CLI の `--no-explain` で無効化） |
| `max_samples` | 2000 | SHAP を計算する最大行数（OOF の行から無作為抽出） |
| `dependence_top_k` | 6 | 「特徴量の値 vs SHAP 値」の散布図を描く特徴量数 |
| `correlation_top_k` | 15 | SHAP 値同士の相関行列に含める特徴量数 |
| `scatter_matrix_top_k` | 5 | SHAP 値同士の散布図行列に含める特徴量数 |

各 fold のモデルを、そのモデルが学習に使っていない検証データで説明します（OOF SHAP）。
explainer はモデルによって変わります。

| モデル | explainer | SHAP 値の尺度 |
|---|---|---|
| 木モデル | `TreeExplainer` | モデルの生出力（分類は log-odds） |
| それ以外 | permutation 法 | 予測値（分類は確率） |

### `evaluation`

| キー | 既定 | 内容 |
|---|---|---|
| `enabled` | true | 誤差評価の図・表を作るか |
| `max_points` | 5000 | 散布図の最大点数 |
| `training_history` | true | 木の本数・エポックごとの学習・検証の損失を記録・描画する |
| `learning_curve` | `{enabled: false, train_sizes: [0.2, …, 1.0]}` | 学習曲線（fold ごとに学習し直すため既定は無効） |
| `validation_curve` | `{param: null, values: []}` | 検証曲線（例: `{param: learning_rate, values: [0.01, 0.05, 0.2]}`） |
| `max_series` | 4 | 複数系列のとき、系列ごとの図を描く最大系列数（検定の表は全系列） |
| `ljung_box_lags` | 自動 | 時系列の残差診断の Ljung-Box 検定のラグ |
| `unit_root_regression` | `c` | 単位根検定（ADF・KPSS）の確定項（`c` / `ct`） |
| `test_alpha` | 0.05 | 検定の判定の有意水準 |

作られる図・表は [evaluation.md](evaluation.md#実験から自動で作られるもの) を参照してください。

### `tracking`

| キー | 既定 | 内容 |
|---|---|---|
| `enabled` | true | MLflow に記録するか（CLI の `--no-tracking` で無効化） |
| `experiment_name` | `name` | MLflow の実験名 |
| `tracking_uri` | `sqlite:///mlruns/mlflow.db` | 記録先 |

### `forecast`（再帰的多段予測、`task: time_series` のみ）

```yaml
task: time_series
data: {train_path: ..., test_path: ..., target: y, time_col: date, drop_cols: [date]}
cv: {method: time_cutoff, cutoffs: ["2024-03-01", "2024-04-01"]}
forecast:
  series_col: series_id        # 複数系列の場合（1つのモデルで全系列を学習する）
  lags: [1, 2, 7]              # y_lag_1, y_lag_2, y_lag_7
  rolling_windows: [7]         # y_lag_1_ma_7（t-1 から過去7期の平均）
  rate_of_change: false        # y_lag_1_roc_1（前期比）
  horizon: 28                  # バックテストで評価する最大ステップ（省略時は検証期間全体）
  clip: {min: 0}               # 予測値の範囲制限
  target_transform: seasonal_diff  # log / diff / log_diff / seasonal_diff / log_seasonal_diff
  seasonal_period: 365         # 季節差分の周期（seasonal 系で必須）
  target_offset: 0             # 対数系の変換で log(y + offset)
```

予測と評価の仕組み:
- **学習:** 真の過去値から作ったラグで「1期先」を予測するモデルを学習します。
- **予測:** 予測値を次の時点の履歴に加え、ラグ・移動平均を作り直しながら1ステップずつ進みます。
- **評価（再帰バックテスト）:** 各 fold の検証期間を、検証期間の実測値を一切使わずに再帰予測します。
  - OOF・チューニング・アンサンブルは、このスコアを基準にします。
  - 参考として、真のラグを使う1期先予測のスコア（`onestep_oof_*`）も記録します。
  - ステップ別の誤差は `horizon_scores.csv` / `horizon_error.png` に出力されます。
- **目的変数の変換（`target_transform`）:** 指定すると、モデルは変換後の系列（例: 前年同日との差）を学習・予測します。予測値は元の尺度に戻してから評価・出力します。
  GBDT は学習した範囲の外へ外挿できないため、トレンドや強い季節性のある系列で有効です。
- **外生変数:** 日付から作る月・曜日などは、通常どおり `features` に書きます。

前提:
- 各系列の行は一定間隔（1行=1ステップ）であること。
- テストデータの外生変数は、予測時点で既知であること。

## タスクと評価指標

| タスク | 目的変数 | 予測値（`oof_pred`） |
|---|---|---|
| `binary` | 2クラス（ラベルは自動でエンコード） | 陽性クラスの確率 (n,) |
| `multiclass` | 3クラス以上 | 各クラスの確率 (n, K) |
| `regression` | 数値 | 予測値 (n,) |
| `time_series` | 数値 | 予測値 (n,)。CV に時間順を強制する点だけが回帰と違う |

| 指標 | タスク | 向き |
|---|---|---|
| `rmse` / `mse` / `mae` / `mape` / `rmsle` / `r2` | 回帰・時系列 | `r2` のみ最大化 |
| `logloss` | 分類 | 最小化 |
| `roc_auc`（別名 `auc`）/ `roc_auc_macro` / `roc_auc_micro` | 分類 | 最大化 |
| `pr_auc` / `pauc`（`pauc@0.05` で FPR の上限を指定。既定 0.1）/ `precision` / `recall` / `f1` | 二値 | 最大化 |
| `accuracy` / `mcc` / `g_mean` / `precision_*` / `recall_*` / `f1_*`（`macro` / `micro` / `weighted`） | 分類 | 最大化 |

- **ラベルを使う指標**（accuracy など）: 二値は確率 0.5 以上を陽性、多クラスは確率が最大のクラスをラベルにして計算します。
- **`rmsle`**: 負の予測値を 0 に切り上げて計算します。
- **一覧の取得**: `modeling.metrics.available_metrics()` で一覧、`get_metric(name)` で `Metric`（`direction` など）を取れます。

## モデル

| `model.name` | 内容 | タスク | 既定の探索空間 |
|---|---|---|---|
| `lightgbm` | `LGBMRegressor` / `LGBMClassifier` | 全タスク | `learning_rate`, `num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| `xgboost` | `XGBRegressor` / `XGBClassifier` | 全タスク | `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| `linear` | 欠損補完＋標準化＋ Ridge / LogisticRegression | 全タスク | `alpha`（回帰）/ `C`（分類） |
| `dummy` | 定数予測（平均・事前確率）。超えるべき最低ラインの確認用 | 全タスク | なし |
| `mlp` | 全結合ネット（PyTorch + skorch） | 全タスク | `lr`, `batch_size`, `module__dropout`, 幅・層数 |
| `cnn1d` | 1次元CNN。特徴量の**並び順を系列とみなす**ので、ラグ特徴量など順序に意味がある入力で使う | 全タスク | `lr`, `batch_size`, `module__dropout`, チャネル数・カーネル幅 |

`mlp` / `cnn1d` は任意依存です。`uv sync --extra nn` で torch・skorch を入れたときだけ登録されます。

### モデルを追加する

`ModelSpec` を継承したクラスを `src/modeling/models/` に置き、`@register_model` を付けます。
`modeling/models/__init__.py` で import すると、YAML の `model.name` で指定できるようになります。

```python
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from modeling.models.base import ModelSpec, merge_params, register_model
from modeling.tasks import Task, is_classification


@register_model
class RandomForestSpec(ModelSpec):
    """ランダムフォレスト。"""

    name = "random_forest"
    supported_tasks = frozenset(Task)
    explainer_kind = "tree"  # SHAP は TreeExplainer を使う

    def build(
        self, task: Task, params: dict[str, Any], seed: int, early_stopping_rounds: int | None = None
    ) -> BaseEstimator:
        merged = merge_params({"random_state": seed, "n_jobs": -1}, params)
        cls = RandomForestClassifier if is_classification(task) else RandomForestRegressor
        return cls(**merged)

    def search_space(self, trial: Any, task: Task) -> dict[str, Any]:
        return {"max_depth": trial.suggest_int("max_depth", 3, 20)}
```

必要に応じて、ほかのメソッドも上書きします。

| メソッド | 用途 |
|---|---|
| `fit_kwargs` | early stopping 用の検証データの渡し方 |
| `best_iteration` / `with_n_iterations` | `refit_full` のときの反復回数 |
| `training_history` | 学習の推移の図 |

## アンサンブル（`EnsembleConfig`）

各実験が保存した OOF 予測・テスト予測を統合します（モデルの再学習はしません）。

```yaml
name: blend
task: regression
members:
  - {name: lgbm, experiment: example_lgbm}         # outputs/experiments/example_lgbm/ の最新の実行
  - {name: xgb, path: outputs/experiments/x/20260101_000000_000000}  # ディレクトリを直接指定
  - {name: nn, run_id: 0123abcd...}                # MLflow の run の artifact から取得
method: weighted      # mean / rank_mean（二値のみ）/ weighted / stacking
metrics: [rmse, mae]
stacking: {model: {name: linear}}   # method: stacking のメタモデル
```

- **`weighted`:** OOF で主指標が最良になる重み（非負・合計1）を探索します。
- **`stacking`:** 構成要素の予測を特徴量にしたメタモデルを学習します。
- **楽観的なスコアの回避:** `weighted` / `stacking` を同じ OOF で評価すると、スコアが楽観的になります。そのため「fold k 以外の OOF で学習して fold k を予測する」cross-fitting で、アンサンブルの OOF スコアを計算します。
- **前提:** 全構成要素が、同じ学習データ・同じ CV 分割で作られていることです。行数・fold 番号・目的変数が一致しなければエラーにします。

Python からは次のように実行します。

```python
from modeling.config import load_ensemble_config
from modeling.ensemble import run_ensemble

result = run_ensemble(load_ensemble_config(path))
print(result.scores)
print(result.weights)
```

## 出力ファイル

`outputs/experiments/{name}/{YYYYmmdd_HHMMSS_ffffff}/`:

| ファイル | 内容 |
|---|---|
| `config.yaml` | 実行した設定（`resolved_params`: 実際に使ったパラメータ） |
| `cv_scores.csv` | fold 別スコア（最終行 `oof` は OOF 全体のスコア） |
| `oof_predictions.parquet` | OOF 予測（列: `row`, `id`, `fold`（どの fold にも入らない行は -1）, `target`, `pred` または `pred_0`…） |
| `test_predictions.parquet` | テスト予測（`test_path` 指定時） |
| `tuning_trials.csv` | Optuna の試行一覧（チューニング時） |
| `horizon_scores.csv` / `horizon_error.png` | 予測ステップ別のスコア（`forecast` 指定時） |
| `evaluation/` | 誤差評価の図・表（[evaluation.md](evaluation.md#実験から自動で作られるもの)） |
| `shap/` | `shap_importance.csv`、`shap_values.parquet`、重要度・beeswarm・相関の図（多クラスは `_class_{k}`） |

`outputs/ensembles/{name}/{日時}/` には、次のファイルが保存されます。

| ファイル | 内容 |
|---|---|
| `config.yaml` | 実行した設定 |
| `ensemble_scores.csv` | 構成要素とアンサンブルの OOF スコア |
| `weights.csv` | 構成要素の重み（`mean` / `weighted` のみ） |
| `oof_predictions.parquet` / `test_predictions.parquet` | アンサンブルの予測 |

MLflow には、次のものが記録されます。

| 種類 | 内容 |
|---|---|
| パラメータ | 設定一式、`resolved_params`、行数・特徴量数 |
| 指標 | `fold_{指標}`（step=fold）、`cv_mean_*` / `cv_std_*` / `oof_*`、`onestep_oof_*`、`tuning_best_*` |
| 成果物 | 上のファイル一式 |
| タグ | git の commit hash |

## 部品として使う API

`run_experiment` を通さずに、一部だけを使うこともできます。

| 関数 | 内容 |
|---|---|
| `experiment.prepare_dataset(config, train, test=None)` | `Dataset`（`X`, `y`, `folds`, `ids`, `X_test` など）を作る |
| `dataset.cross_validate(config, dataset, params=None)` | CV学習（`forecast` 指定時は再帰バックテスト）→ `CVResult` |
| `trainer.run_cv(config, X, y, folds, X_test=None, ...)` | 通常の CV学習ループ |
| `trainer.fit_pipeline(config, X_train, y_train, X_valid=None, y_valid=None)` | Pipeline を1つ学習する |
| `pipeline.build_pipeline(config, params=None)` | 未学習の Pipeline（特徴量ステップ → `ToModelInput` → モデル） |
| `forecasting.recursive_forecast(pipeline, history, future, builder, feature_columns)` | 学習済み Pipeline で未来の行を再帰予測する |
| `tuning.tune(config, dataset, n_trials=None)` | チューニングだけを行う → `TuningResult`（`best_params`, `best_value`, `trials`） |
| `explain.compute_oof_shap(config, X, folds, cv_result)` | OOF SHAP → `ShapResult`（`values`, `data`, `importance()`） |
| `evaluation.save_evaluation_outputs(config, dataset, cv_result, output_dir)` | 誤差評価の図・表だけを作る |
| `cv.make_folds(cfg, frame, y, groups=None, time_column=None)` | CV分割（`(学習の行番号, 検証の行番号)` のリスト） |
| `tasks.predict(estimator, X, task, n_classes=None)` | タスクに応じた予測値（分類は確率） |
| `io.predictions_to_frame` / `frame_to_predictions` | 予測の配列 ⇔ 保存用の DataFrame |

`CVResult` の主な属性:

| 属性 | 内容 |
|---|---|
| `oof_pred` | OOF 予測。どの fold にも入らない行は NaN |
| `fold_ids` | 各行が検証データになった fold 番号 |
| `fold_scores` / `oof_scores` | fold 別のスコア / OOF 全体のスコア |
| `test_pred` | テスト予測 |
| `models` | fold ごとの学習済み Pipeline |
| `best_iterations` | fold ごとの early stopping の最良反復 |
| `params` | 学習に使ったパラメータ |

再帰予測では `ForecastCVResult` になり、`steps`・`horizon_scores`・`onestep_oof_pred`・`onestep_oof_scores` が加わります。

## 注意点

- **リーク防止:** 特徴量のステップ（ターゲットエンコーディングなどを含む）は、fold ごとに学習データだけで fit します。CV の外で特徴量を作るときは、自分でリークに注意してください。
- **`early_stopping_rounds` と OOF:** 検証 fold で打ち切り回数を決めるため、OOF スコアはわずかに楽観的になります。
- **アンサンブルの前提:** 構成要素の実験は、CV の設定（`cv` と `seed`）と学習データをそろえてください。
