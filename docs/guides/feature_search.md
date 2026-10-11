# 特徴量の探索（グループの組み合わせ・アブレーション）

単一モデル（LightGBM など）で特徴量の組み合わせを比べるための、探索の設定（YAML）とスクリプト
`scripts/run_feature_search.py` の使い方と仕様です。

- **[使い方](#使い方)**: 実行の流れ・設定の書き方・結果の読み方。
- **[設計仕様](#設計仕様)**: パターンの組み立ての規則・出力・エラー。

関連:
- [feature_exploration.md](feature_exploration.md): 実験設定の `base` / `use` の仕様。1ファイル = 1パターンで書く方法。
- [configs/feature_search/README.md](../../configs/feature_search/README.md): 同梱の探索の設定の一覧。

## 何ができるか

`configs/experiments/fe_*.yaml` のようにパターンごとに実験設定を書く方法（[feature_exploration.md](feature_exploration.md)）は、
自由度が高い反面、パターンが増えると「どの列を足したか」が分かりにくくなります。比べるのも手作業です。

特徴量の探索では、特徴量を**グループ**（一緒に足し引きする列のまとまり）に分け、グループの組み合わせを**パターン**として並べます。

| 機能 | 内容 |
|---|---|
| パターンの組み立て | 共通の実験設定 ＋ グループの列（`data.feature_cols`）・ステップ（`features`）から、パターンごとの実験設定を作る |
| アブレーション | 指定したパターンからグループを1つずつ外したパターンを自動で作る（どのグループが効いているかが分かる） |
| 比較の前提の確認 | 全パターンが同じデータ・同じ CV 分割・同じ指標で評価されることを、学習の前に確認する |
| 比較 | 基準のパターンとの差を、fold 平均と「基準に勝った fold の数」で示す。特徴量の重要度もパターンごとに集計する |
| 重複の省略 | 設定がまったく同じパターン（アブレーションの結果が別のパターンと同じ列になる場合など）は1回だけ学習する |

---

# 使い方

## 実行の流れ

熱交換器のダミーデータ（前処理済み）の例です。

```bash
# 1. データセットを作る（data/processed/dummy/{train,test}_dataset.parquet）
uv run python scripts/preprocess_heat_exchanger_dummy.py

# 2. 設定の確認（パターンごとのグループ・列数を表示する。学習はしない）
uv run python scripts/run_feature_search.py --config configs/feature_search/hx_dummy.yaml --check

# 3. 実行（全パターン ＋ アブレーション）
uv run python scripts/run_feature_search.py --config configs/feature_search/hx_dummy.yaml
```

同じ内容をセルごとに実行できるノートブック版は `notebook/005_feature_search_heat_exchanger_dummy.ipynb` です
（papermill のパラメータ `CONFIG_PATH`・`PATTERNS`・`ABLATION`・`TRACKING`・`OUTPUT_ROOT` がスクリプトの引数に対応します）。

| オプション | 内容 |
|---|---|
| `--patterns <名前> ...` | 指定したパターンだけ実行する（基準のパターンとアブレーションの土台は、比較のために自動で含める） |
| `--no-ablation` | アブレーションを行わない |
| `--no-tracking` | MLflow に記録しない |
| `--output-root <dir>` | 出力のルート（既定: `outputs/feature_search`） |
| `--check` | 設定の検証と組み立て結果の表示だけを行う |

## ファイルの置き場所

```
configs/
├── feature_search/
│   ├── hx_dummy.yaml              # 探索の設定（グループ・パターン・アブレーション）
│   └── hx_dummy/
│       └── experiment.yaml        # 全パターン共通の実験設定（データ・CV・モデル・指標）
└── features/
    ├── hx_chemical_ordinal.yaml   # グループから use で呼ぶ特徴量ブロック
    └── hx_pressure_drop_lags.yaml
```

共通の実験設定は `configs/experiments/` に置きません。
`run_experiment.py --config "configs/experiments/*.yaml"` に、`name` の無い部品のファイルが混ざらないようにするためです。

## 1. 共通の実験設定

`ExperimentConfig`（[modeling.md](../modules/modeling.md#設定リファレンスexperimentconfig)）から、`name` と `data.feature_cols` を除いたものです。
`base` での継承や、`features`（全パターンに入るステップ）も書けます。

```yaml
# configs/feature_search/hx_dummy/experiment.yaml
task: regression
data:
  train_path: data/processed/dummy/train_dataset.parquet
  test_path: data/processed/dummy/test_dataset.parquet
  target: 汚れ係数
  time_col: 日付
cv:
  method: time_cutoff
  cutoffs: ["2022-01-01", "2023-01-01", "2024-01-01"]
  valid_end: "2025-01-01"
model:
  name: lightgbm
  params: {n_estimators: 3000, learning_rate: 0.05, num_leaves: 31}
  early_stopping_rounds: 100
metrics: [rmse, mae]          # 先頭が比較に使う主指標
explain: {enabled: false}     # 探索を速く回すため SHAP・誤差評価は無効
evaluation: {enabled: false}
```

## 2. 探索の設定

```yaml
# configs/feature_search/hx_dummy.yaml
name: hx_dummy_fe
experiment: hx_dummy/experiment.yaml   # このファイルからの相対パス

groups:
  unit_calendar: [機器番号, 月]          # 省略形: 列のリストだけ
  sensors: [入口圧力, 出口圧力, 高温側流量, 低温側流量]
  chemical_type:                         # 列と一緒にステップも加える形
    description: 薬剤種類（文字列なので整数に置き換える）
    columns: [薬剤種類]
    features: [{use: hx_chemical_ordinal}]
  pressure_drop: [圧力損失, 圧力損失_ma_7]
  # ...

patterns:
  - name: 01_raw
    groups: [unit_calendar, sensors, crude_ratio, chemical_type]
  - name: 02_pressure_drop
    groups: [unit_calendar, sensors, crude_ratio, chemical_type, pressure_drop]
  - name: 08_all_with_lags
    groups: ["*"]                        # 全グループ
  - name: 10_compact_slow_lr
    groups: [unit_calendar, pressure_drop, cleaning_cycle, pressure_rise]
    overrides:                           # 共通の実験設定への上書き
      model: {params: {learning_rate: 0.02}}

baseline: 01_raw                         # 差を測る基準（省略時は patterns の先頭）

ablation:
  pattern: 07_all_preprocessed           # このパターンからグループを1つずつ外す
  keep: [unit_calendar]                  # 外さないグループ
```

## よくある書き方

### 前処理で作った列の効果を順に確かめる

元データの列だけのパターンを基準にし、グループを1つずつ足したパターンを並べます（同梱の `hx_dummy.yaml` の `01`〜`07`）。

### どのグループが効いているかを見る

全部入りのパターンを `ablation.pattern` に指定します。`ablation.csv` の `loss_when_removed` が大きいグループほど効いています。
負のグループは、外した方がスコアが良くなっています。

### CV の中で作る特徴量を試す

ラグ・移動平均・ターゲットエンコーディングのように、学習データから作る特徴量はグループの `features` に書きます。
CV の fold ごとに Pipeline の中で作られるため、検証データの情報は学習に混ざりません。
ステップが使う列（例: `group_by` の `機器番号`）は、そのグループの `columns` にも入れてください。

```yaml
groups:
  pressure_lags:
    columns: [機器番号, 圧力損失]
    features: [{use: hx_pressure_drop_lags}]
```

### パラメータを変えて比べる

特徴量が同じでも、モデルのパラメータで結果が変わります。`overrides` でパラメータを変えたパターンを加えます。
`overrides` は dict 単位でマージされるので、書いたキーだけが変わります。

### 一部のパターンだけ実行し直す

```bash
uv run python scripts/run_feature_search.py --config configs/feature_search/hx_dummy.yaml --patterns 05_cleaning_cycle --no-ablation
```

基準のパターン（`01_raw`）は比較のために自動で含まれます。

## 結果の読み方

出力は `outputs/feature_search/<name>/<日時>/` にまとまります。

| ファイル | 内容 |
|---|---|
| `summary.csv` ・ `summary.png` | パターンごとのスコア（fold 平均・標準偏差・OOF）、基準との差、基準に勝った fold の数、順位 |
| `fold_scores.csv` ・ `fold_scores.png` | パターン × fold のスコア |
| `ablation.csv` ・ `ablation.png` | 外したグループと、外したときに悪くなった量（`loss_when_removed`） |
| `feature_importance.csv` | パターン × 特徴量の重要度（fold 平均。LightGBM は gain の割合） |
| `search.yaml` | 探索の設定と、各パターンのグループ・列（再現用） |
| `experiments/<name>__<パターン>/<日時>/` | 各パターンの実験の出力（`cv_scores.csv`・予測など） |

読むときの注意:

- **差が小さいとき**: `improvement_<主指標>` が `cv_std_<主指標>` より小さい差は、偶然の範囲として扱います。
  `fold_wins` が全 fold で勝っているかもあわせて確認してください。
- **探索した CV で選んでいる**: 多くのパターンから最良を選ぶと、その CV スコアは楽観的になります。
  最終的な評価は、探索に使っていない期間（テスト期間など）で確かめてください。
- **early stopping**: 検証 fold で木の本数を決めるため、スコアはわずかに楽観的です（全パターン同じ条件なので比較には使えます）。
- 有望なパターンは、`overrides: {explain: {enabled: true}, evaluation: {enabled: true}}` のパターンを加えて実行し、SHAP と誤差を詳しく確認します。

MLflow には、実験名 `<name>`（共通の実験設定の `tracking.experiment_name` で変更可）で、パターンごとの run が記録されます。
タグ `feature_search`・`feature_pattern` で絞り込めます。

---

# 設計仕様

## 設定のスキーマ（`modeling.config.FeatureSearchConfig`）

| キー | 型 | 既定 | 内容 |
|---|---|---|---|
| `name` | str | 必須 | 探索名（英数字・`_`・`-`・`.`・`+`）。出力ディレクトリ名と、MLflow の既定の実験名 |
| `experiment` | パス or mapping | 必須 | 共通の実験設定。パスは探索の YAML からの相対パスで、`base`・`use` も展開される。`name` は書けない |
| `groups` | mapping | 必須 | グループ名 → `{columns, features, description}`、または列のリスト（省略形） |
| `patterns` | list | 必須 | `{name, groups, features, overrides, description}` のリスト |
| `baseline` | str | 先頭のパターン | 差を測る基準のパターン名 |
| `ablation.pattern` | str | なし | アブレーションの土台のパターン名（なしなら行わない） |
| `ablation.keep` | list | `[]` | アブレーションで外さないグループ |

探索の YAML 自体にも `base` を書けます（グループの定義を共有する場合など）。

## パターンの実験設定の組み立て

`modeling.feature_search.build_pattern_config` が、次の順に組み立てます。

1. 共通の実験設定に、パターンの `overrides` を再帰的にマージする（dict はキー単位、リストは置き換え）。
2. `name` を `<探索名>__<パターン名>` にする。
3. `data.feature_cols` を、パターンのグループの `columns` をグループの順につなげたものにする（重複は除く）。
4. `features` を「共通の実験設定のステップ → グループのステップ（グループの順）→ パターンのステップ」にする。
5. `tracking.experiment_name` が無ければ探索名にする。
6. `ExperimentConfig` として検証する。

`groups: ["*"]` は、定義済みの全グループ（定義順）に展開されます。

アブレーションのパターンは、土台のパターンのグループから1つを除いて同じ手順で組み立てます。
名前は `<土台>__minus_<グループ>` です。土台の `overrides`・`features` は引き継ぎます。

## 比較の前提の確認

学習の前に次を確認し、満たさなければエラーにします。

| 確認 | エラー |
|---|---|
| `overrides` で `task`・`cv`・`metrics`・`features`・`name`、`data` の `train_path`・`test_path`・`target`・`id_col`・`group_col`・`time_col`・`feature_cols` を変えていない | `overrides では [...] を変えられません` |
| 全パターンの task・data（上記の項目）・cv・metrics・forecast が同じ | `パターン X の [...] が Y と異なります` |
| パターンが使う列がすべて学習データにある | `パターン X: 学習データに無い列 [...]` |

設定の読み込み時には、未定義のグループ・パターン名の重複・`baseline` や `ablation` の参照先の誤り、
列もステップも無いグループをエラーにします。

## 実行

`modeling.feature_search.run_feature_search` は、学習データ（とテストデータ）を1回だけ読み込みます。
パターンごとに `prepare_dataset` で列を選び、`run_experiment` で CV 学習します。

- `name`・`tracking` を除いた実験設定がまったく同じパターンは、最初の1回の結果を使い回します（列の順序も含めて比べる）。
- MLflow の run には、タグ `feature_search`（探索名）・`feature_pattern`（パターン名）を付けます。

## 集計

| 列（`summary.csv`） | 内容 |
|---|---|
| `n_columns` / `n_model_features` | パターンの列数 / モデルに渡った特徴量の数（Pipeline の中で作った列を含む） |
| `cv_mean_*` ・ `cv_std_*` ・ `oof_*` | 指標ごとの fold 平均・標準偏差・OOF 全体のスコア |
| `improvement_<主指標>` | fold ごとの「基準より良くなった量」の平均（小さいほど良い指標では符号を反転する。正が良い） |
| `fold_wins` / `n_folds` | 基準より良かった fold の数 / fold の数 |
| `rank` | 主指標の fold 平均の順位（同点は同順位） |

`ablation.csv` の `loss_when_removed` は「土台の improvement − 外したパターンの improvement」で、正なら外すと悪くなります。

特徴量の重要度は、fold ごとのモデルの重要度を合計 1 に正規化してから平均します。
LightGBM は gain、それ以外は `feature_importances_` を使います（無いモデルでは出力しません）。

## 実装

| 対象 | 場所 |
|---|---|
| 設定のスキーマ・読み込み | `src/modeling/config.py` の `FeatureSearchConfig` ・ `load_feature_search_config` |
| 組み立て・実行・集計・図 | `src/modeling/feature_search.py` |
| CLI | `scripts/run_feature_search.py` |
| テスト | `tests/test_modeling_feature_search.py` |
