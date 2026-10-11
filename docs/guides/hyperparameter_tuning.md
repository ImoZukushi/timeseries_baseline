# ハイパーパラメータの探索とアンサンブル

各モデルのハイパーパラメータを Optuna で探索し、最良のパラメータで学習したモデルをアンサンブルして最終モデルを作るまでを、YAML とスクリプト1本で自動化する方法です。

- **[使い方](#使い方)**: 全体の流れ・設定の書き方・実行・結果の見方・よくある書き方。
- **[設計仕様](#設計仕様)**: 試行回数と study の扱い・事前の検証・出力・アンサンブルの評価方法。

関連: [modeling.md](../modules/modeling.md)（実験設定の全項目）／ [feature_exploration.md](feature_exploration.md)（設定の継承 `base` と特徴量ブロック `use`）

## 全体の流れ

```
configs/pipelines/<名前>.yaml（パイプライン設定）
  │  members: 構成モデルの実験設定（探索範囲は configs/search_spaces/ から base で継承）
  ▼
scripts/run_pipeline.py
  0. 全構成モデルの設定を読み込み、アンサンブルできる組み合わせかを検証（学習の前）
  1. 構成モデルごとに Optuna で探索（study 全体で n_trials 回に達するまで）
  2. 最良パラメータで CV 学習 → OOF 予測・テスト予測を保存
  3. 指定した統合方法（mean / weighted / stacking など）ごとにアンサンブル
  4. 構成モデル単体と各アンサンブルの OOF スコアを比べ、最良を「最終モデル」とする
  ▼
outputs/pipelines/<名前>/<日時>/  summary.csv（比較表）・final.yaml（最終モデル）など
```

---

# 使い方

## ファイルの置き場所

```
configs/
├── search_spaces/              # モデルごとの探索範囲（実験設定から base で継承）
│   ├── lightgbm.yaml
│   ├── xgboost.yaml
│   └── linear.yaml
└── pipelines/
    ├── example_ensemble.yaml   # パイプライン設定
    └── example_ensemble/       # その構成モデルの実験設定
        ├── common.yaml         # 共通設定（データ・CV・指標）
        ├── lgbm.yaml           # base: [common.yaml, ../../search_spaces/lightgbm.yaml]
        ├── xgb.yaml
        └── linear.yaml
```

構成モデルの実験設定は、`configs/experiments/` ではなくパイプラインの下に置きます。
部品のファイル（`common.yaml` など）が、`run_experiment.py --config "configs/experiments/*.yaml"` に混ざらないようにするためです。

## 1. 探索範囲を書く

`configs/search_spaces/<モデル>.yaml` に、探索しない固定値（`model.params`）と探索範囲（`tuning.search_space`）を書きます。

```yaml
# configs/search_spaces/lightgbm.yaml
model:
  name: lightgbm
  params: {n_estimators: 5000}   # 探索しない固定値。木の本数は early stopping で決める
  early_stopping_rounds: 100
tuning:
  search_space:
    learning_rate: {type: float, low: 0.005, high: 0.2, log: true}
    num_leaves: {type: int, low: 8, high: 256, log: true}
    min_child_samples: {type: int, low: 5, high: 200, log: true}
    subsample: {type: float, low: 0.5, high: 1.0}
    subsample_freq: {type: fixed, value: 1}
    colsample_bytree: {type: float, low: 0.3, high: 1.0}
    reg_alpha: {type: float, low: 1.0e-8, high: 10.0, log: true}
    reg_lambda: {type: float, low: 1.0e-8, high: 10.0, log: true}
```

| `type` | 書き方 | Optuna |
|---|---|---|
| `float` | `{type: float, low, high, log: false, step: null}` | `suggest_float` |
| `int` | `{type: int, low, high, log: false, step: 1}` | `suggest_int` |
| `categorical` | `{type: categorical, choices: [...]}` | `suggest_categorical` |
| `fixed` | `{type: fixed, value: ...}` | 探索せず固定 |

- **既定の探索範囲を使う場合:** `tuning.search_space` を書かなければ、モデルの既定の探索範囲を使います（[modeling.md のモデル一覧](../modules/modeling.md#モデル)）。
  書いた場合は既定を**置き換え**ます（足し合わせません）。
- **固定値との関係:** 探索したパラメータは `model.params` の固定値を上書きします。

## 2. 構成モデルの実験設定を書く

共通設定（データ・CV・指標）と探索範囲を `base` で継承し、名前と特徴量を書きます。

```yaml
# configs/pipelines/example_ensemble/common.yaml
task: regression
data:
  train_path: data/processed/train_with_departure_time.csv
  target: 合計
  group_col: 年月日
  drop_cols: [年月日, 発車時刻, フェンダー部分(東京方向), 台車部分, フェンダー部分(金沢方向)]
cv: {method: group, n_splits: 5, seed: 42}
metrics: [rmse, mae]
tuning: {n_trials: 30}
explain: {enabled: false}     # 探索中は SHAP・誤差評価を省いて速くする
evaluation: {enabled: false}
tracking: {experiment_name: example_ensemble}
```

```yaml
# configs/pipelines/example_ensemble/lgbm.yaml
base: [common.yaml, ../../search_spaces/lightgbm.yaml]
name: example_ensemble_lgbm
features:
  - use: categorical_ordinal
```

構成モデルごとに、特徴量を変えてかまいません。むしろ、性質の違うモデル・特徴量を混ぜる方がアンサンブルの効果が出やすくなります。
例えば `linear.yaml` は、線形モデルに向くターゲットエンコーディング（`categorical_target`）を使っています。

## 3. パイプライン設定を書く

```yaml
# configs/pipelines/example_ensemble.yaml
name: example_ensemble
members:                          # パスはこの YAML からの相対パス
  - config: example_ensemble/lgbm.yaml
    n_trials: 30                  # study 全体での試行回数の目標（省略時は実験設定の tuning.n_trials）
    timeout: 1800                 # 今回の実行での打ち切り秒数（任意）
  - config: example_ensemble/xgb.yaml
    n_trials: 30
  - config: example_ensemble/linear.yaml
    tune: false                   # 探索せず、実験設定の model.params で学習する
ensemble:
  methods: [mean, weighted, stacking]   # すべて実行して比較する
  metrics: null                         # 省略時は構成モデルの metrics（先頭が主指標）
  stacking: {model: {name: linear}}     # stacking のメタモデル
seed: 42
tracking: {experiment_name: example_ensemble}
```

## 4. 実行する

```bash
# 設定の検証だけ（学習しない）
uv run python scripts/run_pipeline.py --config configs/pipelines/example_ensemble.yaml --check

# 一括実行
uv run python scripts/run_pipeline.py --config configs/pipelines/example_ensemble.yaml
```

| オプション | 内容 |
|---|---|
| `--check` | 設定の検証だけを行う |
| `--skip-tuning` | 全構成モデルを探索せずに学習する（パイプラインの動作確認や、探索済みの設定の比較に） |
| `--no-tracking` | MLflow に記録しない |
| `--output-root` | 出力のルート（既定 `outputs/pipelines`） |
| `--optuna-dir` | study の保存先（既定 `outputs/optuna`） |

**中断と再開:**
- 途中で止めても、同じコマンドで再実行すれば各 study の続きから探索します。
- 目標の試行数に達した構成モデルは、探索を飛ばして学習だけを行います（数秒〜数十秒）。

## 5. 結果を見る

`outputs/pipelines/<名前>/<日時>/` に、次のファイルが保存されます。

| ファイル | 内容 |
|---|---|
| `summary.csv` | 構成モデル単体（`kind=member`）と各アンサンブル（`kind=ensemble`）の OOF スコアと、主指標の順位 `rank` |
| `final.yaml` | 主指標が最良のもの（最終モデル）。方法・スコア・重み・テスト予測のパス |
| `members.csv` | 構成モデルごとの study 名・完了試行数・今回の試行数・探索の最良値・CV スコア・出力先 |
| `best_params.yaml` | 構成モデルごとの、学習に使ったパラメータ（探索結果＋固定値） |
| `pipeline.yaml` | 実行したパイプライン設定と、各構成モデルの出力先 |
| `experiments/<構成モデル>/<日時>/` | 各構成モデルの実験の出力。`tuning_trials.csv`（全試行）・`tuning_summary.yaml`（study 名・最良パラメータ）・OOF/テスト予測など |
| `ensembles/<名前>_<方法>/<日時>/` | 統合方法ごとのアンサンブルの出力（`weights.csv`・予測など） |

`summary.csv` の例:

| kind | name | rmse | mae | rank |
|---|---|---|---|---|
| ensemble | weighted | 0.812 | 0.080 | 1 |
| ensemble | stacking | 0.813 | 0.080 | 2 |
| member | example_ensemble_lgbm | 0.815 | 0.081 | 3 |
| … | … | … | … | … |

テスト予測は、構成モデルの実験設定に `data.test_path` があるときだけ作られます。最終モデルのテスト予測は `final.yaml` の `test_predictions` に入っています。

MLflow には、次のように記録されます。

| 対象 | 記録先・内容 |
|---|---|
| 構成モデル | 各実験設定の `tracking`。各 run にタグ `pipeline`・`pipeline_run`・`optuna_study` を付け、試行は子 run になる |
| アンサンブル | パイプライン設定の `tracking` |

## よくある書き方

### 探索範囲を構成モデルごとに少し変える

`tuning.search_space` は dict なので、継承した範囲をパラメータ単位で上書きできます。

```yaml
base: [common.yaml, ../../search_spaces/lightgbm.yaml]
name: example_ensemble_lgbm_small_lr
tuning:
  search_space:
    learning_rate: {type: float, low: 0.005, high: 0.05, log: true}   # この項目だけ変える
    num_leaves: {type: fixed, value: 31}                              # 探索をやめて固定する
```

### 試行回数を増やして探索を続ける

`n_trials` を増やして再実行すると、増えた分だけ探索します（それまでの試行はそのまま使われます）。

### 動作確認だけを速く回す

```yaml
# 試行回数だけ小さくした別のパイプライン（members は土台のファイルからの相対パスのまま使える）
base: example_ensemble.yaml
name: example_ensemble_small
members:
  - {config: example_ensemble/lgbm.yaml, n_trials: 3}
  - {config: example_ensemble/xgb.yaml, n_trials: 3}
  - {config: example_ensemble/linear.yaml, n_trials: 3}
```

`members` はリストなので、`base` から継承すると全体が置き換わります。試行回数だけを変えたいときも、全構成モデルを書きます。

### 構成モデルを1つの実験として回す

構成モデルの実験設定は、通常の実験設定です。1つだけ試すときは `run_experiment.py` でも実行できます。
試行回数と study の扱いは、パイプラインと同じです。

```bash
uv run python scripts/run_experiment.py --config configs/pipelines/example_ensemble/lgbm.yaml --tune --n-trials 30
```

---

# 設計仕様

## 試行回数（`n_trials`）は「study 全体の目標」

- 探索の前に、study 内の終了済みの試行（完了＋打ち切り）を数え、`n_trials` に足りない分だけ実行します。
- 達していれば探索せず、保存済みの最良パラメータで学習します。
- `timeout` は、その回の実行の上限です。時間切れで目標に届かなくても、次の実行で続きから進みます。
- この扱いは、`run_experiment.py --tune --n-trials N` と、実験設定の `tuning.n_trials` でも同じです。

## study は「探索の条件」ごとに分ける

study の名前は `<実験名>__<設定の指紋8桁>` です。保存先は `outputs/optuna/<実験名>.db`（SQLite）です。
指紋は、探索の目的関数（CV スコア）を変える設定のハッシュです。これらが変わると自動で新しい study になり、条件の違う試行が混ざりません。

| 指紋に含める（変わると新しい study） | 含めない（変えても同じ study） |
|---|---|
| `task` | 実験名以外の表示用の設定 |
| `data`（列の指定）と**学習データの内容**（ファイルのハッシュ） | `explain`（SHAP） |
| `features`（展開後のステップ） | `evaluation`（誤差評価） |
| `cv` | `tracking`（MLflow） |
| `model`（`name`・固定の `params`・`early_stopping_rounds`） | `test_prediction` |
| 主指標（`metrics` の先頭） | `tuning.n_trials`・`tuning.timeout` |
| `tuning.search_space`・`tuning.pruning` | |
| `forecast` | |
| `seed`（Optuna のサンプラーとモデルの乱数） | |

- 実験名は study 名の先頭に入るため、実験名を変えても別の study になります。
- 使った study 名は、出力の `tuning_summary.yaml`・`members.csv`・MLflow のタグ `optuna_study` に残ります。
- 古い study は `.db` の中に残ります。

## 探索の目的関数

各試行は、構成モデルの実験設定どおりの CV（`cv` の分割、`forecast` があれば再帰バックテスト）の、主指標の fold 平均を最適化します。

| 項目 | 内容 |
|---|---|
| サンプラー | TPE（`seed` で固定） |
| 打ち切り | `tuning.pruning: true` のとき、fold ごとのスコアで見込みの薄い試行を途中で打ち切る（MedianPruner） |
| early stopping | 有効なとき、木の本数は試行ごとに検証 fold で決まる。`n_estimators` は大きめの固定値にしておく |

## 実行前の検証（アンサンブルの前提）

アンサンブルは、構成モデルの OOF 予測を**行ごとに**組み合わせます。そのため学習を始める前に、全構成モデルで次がそろっているかを確認し、違えばエラーにします。

| 項目 | 理由 |
|---|---|
| `task` | 予測値の形・意味をそろえる |
| `data` の `train_path`・`test_path`・`target`・`id_col`・`group_col`・`time_col` | 同じ行・同じ目的変数で評価する |
| `cv`（方法・分割数・seed など） | 同じ CV 分割にする（OOF の fold が一致する） |
| `forecast` の `series_col`・`horizon` | 再帰予測で評価する行をそろえる |
| 主指標 | 探索・アンサンブルで同じ指標を最適化する |

- **そろえなくてよい項目:** 特徴量（`features`・`drop_cols`）・モデル・探索範囲は、構成モデルごとに変えてかまいません。
- **ほかのエラー:** 次の場合もエラーにします。
  - 構成モデルの `name` が重複している。
  - アンサンブルの指標がタスクに合わない。
  - `rank_mean` を二値分類以外で指定した。

## アンサンブルの評価

- **構成要素の受け渡し:** 各統合方法は、手順2で保存した**今回の実行結果**を、出力ディレクトリのパスで受け取ります（「最新の実行」を探しません）。
- **`weighted` / `stacking` の評価:** OOF で重み・メタモデルを学習するため、同じ OOF で評価すると楽観的になります。そこで「fold k 以外で学習して fold k を予測する」cross-fitting で作ったアンサンブルの OOF で評価します。
- **テスト予測:** 全 OOF で学習した重み・メタモデルで作ります。
- **`summary.csv` のスコア:** 構成モデル単体とアンサンブルを、同じ行（OOF 予測のある行）で計算した値です。
- **最終モデル:** `summary.csv` で主指標が最良のものです。アンサンブルより単体のモデルが良ければ、単体が選ばれます。

| 方法 | 内容 | 向いている場面 |
|---|---|---|
| `mean` | 単純平均 | 構成モデルの性能が近いとき。過学習しにくい |
| `weighted` | OOF で主指標が最良になる重み（非負・合計1） | 性能に差があるとき |
| `stacking` | 構成モデルの予測を特徴量にしたメタモデル（既定は線形） | 構成モデルが多く、得意な領域が違うとき |
| `rank_mean` | 順位の平均（二値分類のみ） | AUC など順位だけで評価するとき |

## 注意点

| 注意点 | 内容 |
|---|---|
| CV スコアは楽観的 | 探索で選んだパラメータを同じ CV で評価するため、本番の性能より良く見えがち。試行回数が多いほど強まる。最終確認には、探索に使っていない期間・データで評価する |
| CV の差の大きさ | fold 間のばらつき（`members.csv` の `cv_std_*`）より小さい差は、偶然の範囲として扱う |
| 計算時間 | 各試行で CV（fold 数ぶんの学習）を行う。`timeout` と pruning で抑え、探索中は SHAP・誤差評価を無効にしておく |
| 多様性 | 同じモデル・同じ特徴量の構成モデルを増やしても、アンサンブルの効果は小さい。モデルの種類・特徴量を変える |
| study の掃除 | 条件を変えるたびに新しい study ができる。不要になったら `outputs/optuna/<実験名>.db` を消してよい（次の実行は新しく探索する） |
| Windows のパスの長さ | 出力は `outputs/pipelines/<名前>/<日時>/experiments/<構成モデル>/<日時>/...` と深くなる。`--output-root` を深い場所にすると、パスが 260 文字を超えて保存に失敗することがある |

## 実装

| 対象 | 場所 |
|---|---|
| パイプライン設定 | `src/modeling/config.py`: `TuningPipelineConfig`・`PipelineMemberConfig`・`PipelineEnsembleConfig`・`load_pipeline_config` |
| 実行・事前検証・比較表 | `src/modeling/tuning_pipeline.py`: `run_tuning_pipeline`・`load_members`・`validate_members` |
| 探索（目標回数・study の指紋） | `src/modeling/tuning.py`: `tune`・`study_name`・`study_fingerprint` |
| 構成モデルの学習 | `src/modeling/experiment.py`: `run_experiment`（`timeout`・`extra_tags`・`tuning_summary.yaml`） |
| アンサンブル | `src/modeling/ensemble.py`: `run_ensemble` |
| CLI | `scripts/run_pipeline.py` |
| テスト | `tests/test_modeling_tuning.py`・`tests/test_modeling_tuning_pipeline.py` |
