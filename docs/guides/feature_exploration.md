# 特徴量の探索: 設定の継承（`base`）と特徴量ブロック（`use`）

特徴量エンジニアリングをいろいろなパターンで試すための、実験設定（YAML）の書き方の仕様と使い方です。

- **[使い方](#使い方)**: 探索の流れ・よくある書き方（レシピ）・結果の比較。
- **[設計仕様](#設計仕様)**: 展開の正確な規則・エラー・制約。

特徴量を列のグループに分けて、組み合わせやアブレーションを一括で比べたい場合は [feature_search.md](feature_search.md) を使います。

関連: [modeling.md](../modules/modeling.md)（実験設定の全項目）／ [feature_engineering.md](../modules/feature_engineering.md)（使える transformer）／ [configs/features/README.md](../../configs/features/README.md)（ブロックの一覧）

## なぜ必要か

実験の YAML は1ファイルで完結しているので、特徴量だけを変えるときも、データ・CV・モデル・指標をすべて写す必要がありました。
パターンが増えると、共通部分を直し忘れて CV分割やモデル設定がずれ、比較が成り立たなくなります。

そこで YAML の読み込み時に、次の2つを展開するようにしました。

| 仕組み | 書き方 | 役割 |
|---|---|---|
| 設定の継承 | `base: fe_hx_base.yaml` | 共通部分を1か所に置き、各パターンは**差分だけ**を書く |
| 特徴量ブロック | `features: [{use: hx_pressure_drop_lags}]` | 特徴量ステップのまとまりに名前を付け、**部品として差し替える** |

展開は、検証（pydantic）の前に行います。展開後は従来と同じ設定になるので、学習・予測・出力の処理は何も変わりません。

---

# 使い方

## ファイルの置き場所

```
configs/
├── experiments/
│   ├── common/
│   │   ├── hx_dummy.yaml          # データ・特徴量・CV・指標の共通設定（name なし）
│   │   └── lgbm.yaml              # LightGBM の設定
│   ├── hx_lgbm.yaml               # base: [common/hx_dummy.yaml, common/lgbm.yaml]
│   ├── fe_hx_base.yaml            # 探索の土台（base: hx_lgbm.yaml ＋ SHAP・誤差評価なし）
│   ├── fe_hx_pressure_lags.yaml   # base: fe_hx_base.yaml ＋ 差分
│   └── fe_hx_no_flags.yaml        # base: fe_hx_base.yaml ＋ 差分
└── features/                      # 特徴量ブロック（use: <ファイル名> で呼ぶ）
    ├── hx_chemical_ordinal.yaml
    └── hx_pressure_drop_lags.yaml
```

`common/` のようなサブディレクトリの設定は、`configs/experiments/*.yaml` の glob に含まれません。
`name` の無い部品は、サブディレクトリに置いてください。

## 探索の流れ

### 1. 共通設定（ベースライン）を用意する

`configs/experiments/common/hx_dummy.yaml` に、全パターンで共通のデータ・特徴量・CV・指標を書きます。

```yaml
task: regression
data:
  train_path: data/processed/dummy/train_dataset.parquet
  test_path: data/processed/dummy/test_dataset.parquet
  target: 汚れ係数
  time_col: 日付
  drop_cols: [日付, 機器]
features:
  - use: hx_chemical_ordinal
cv:
  method: time_cutoff
  cutoffs: ["2022-01-01", "2023-01-01", "2024-01-01"]
  valid_end: "2025-01-01"
metrics: [rmse, mae]
```

探索の土台 `fe_hx_base.yaml` は、モデルの例 `hx_lgbm.yaml` を継承し、探索を速く回すための設定だけを足します。

```yaml
base: hx_lgbm.yaml
name: fe_hx_base
explain: {enabled: false}       # SHAP を作らない
evaluation: {enabled: false}    # 誤差評価を作らない
tracking: {experiment_name: feature_exploration_hx}   # MLflow の実験名をまとめる
```

### 2. 試したい特徴量をブロックにする

`configs/features/<名前>.yaml` に、`steps`（`features` と同じ書式のリスト）を書きます。

```yaml
# configs/features/hx_pressure_drop_lags.yaml
description: 圧力損失のラグ（1・7・14行前）と30行の移動平均（機器ごと）
steps:
  - class: feature_engineering.time_series.LagFeatureGenerator
    params: {variables: [圧力損失], lags: [1, 7, 14], group_by: 機器番号}
  - class: feature_engineering.time_series.MovingAverageTransformer
    params: {variables: [圧力損失], window: 30, min_periods: 10, group_by: 機器番号}
```

### 3. パターンごとの YAML を差分だけで書く

```yaml
# configs/experiments/fe_hx_pressure_lags.yaml
base: fe_hx_base.yaml
name: fe_hx_pressure_lags
features:
  - use: hx_chemical_ordinal     # 土台の分（リストは置き換えなので書き直す）
  - use: hx_pressure_drop_lags   # 追加分
```

`name` は必ずパターンごとに変えてください。出力ディレクトリ名と MLflow の run 名になります。

### 4. 展開結果を確認する（任意）

継承と展開の結果が意図どおりかは、次のコマンドで確認できます。

```bash
uv run python -c "import sys, yaml; sys.path.insert(0, 'src'); from pathlib import Path; from modeling.config import load_config_dict; print(yaml.safe_dump(load_config_dict(Path('configs/experiments/fe_hx_pressure_lags.yaml')), allow_unicode=True, sort_keys=False))"
```

### 5. まとめて実行する

```bash
uv run python scripts/run_experiment.py --config "configs/experiments/fe_hx_*.yaml"
```

- 実行前に、すべての設定を展開・検証します。ブロック名の誤りなどは、学習を始める前にエラーになります。
- 各パターンの出力は `outputs/experiments/<name>/<日時>/` に保存されます。
  出力の `config.yaml` には**展開済みの設定**（実際に使ったステップの一覧）が残ります。

### 6. 比較する

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
```

MLflow の実験 `feature_exploration_hx` で、各 run の指標を並べます。

| 指標 | 内容 |
|---|---|
| `cv_mean_<指標>` | fold 平均のスコア |
| `cv_std_<指標>` | fold 間のばらつき |
| `fold_<指標>` | fold ごとのスコア（step = fold 番号） |

ファイルで見る場合は、各出力の `cv_scores.csv`（最終行 `oof` が OOF 全体）を並べます。

全パターンが同じデータ・同じ CV 分割（`cv` と `seed`）を使うので、スコアの差は特徴量の違いによるものです。
ただし、差が `cv_std_*` より小さい場合は偶然の範囲として扱い、fold ごとの勝ち負けもあわせて確認してください。

### 7. 有望なパターンを深掘りする

有望なパターンだけ、SHAP と誤差評価を有効にして実行し直します。

```yaml
# configs/experiments/fe_hx_pressure_lags_detail.yaml
base: fe_hx_pressure_lags.yaml
name: fe_hx_pressure_lags_detail
explain: {enabled: true}
evaluation: {enabled: true}
```

## レシピ

### 特徴量を足す（ベースラインに追加）

リストは**置き換え**です。土台の `features` に足すときは、土台のブロックも書き直します（`fe_hx_pressure_lags.yaml`）。

```yaml
base: fe_hx_base.yaml
name: fe_hx_pressure_lags
features:
  - use: hx_chemical_ordinal     # 土台の分
  - use: hx_pressure_drop_lags   # 追加分
```

### 特徴量を使わない列を変える

`data` は dict なので、キー単位で上書きされます。`drop_cols` だけを書けば、`train_path` などは土台のままです。
`drop_cols` 自体はリストなので、残したい要素を全部書きます（`fe_hx_no_flags.yaml`）。

```yaml
base: fe_hx_base.yaml
name: fe_hx_no_flags
data:
  drop_cols: [日付, 機器, 入口圧力_外れ値, 出口圧力_外れ値, ...]   # 土台の 日付・機器 も書く
```

使う列を明示したい場合は、`data.feature_cols` に列を並べます（`drop_cols` より優先されます）。

### ブロックを使わずに1ステップだけ試す

通常のステップと `use` は混ぜて書けます。

```yaml
base: fe_hx_base.yaml
name: fe_hx_log_flow
features:
  - use: hx_chemical_ordinal
  - class: feature_engineering.numeric.LogTransformer
    params: {variables: [高温側流量, 低温側流量], offset: 1}
```

### ハイパーパラメータを1つだけ変える

`model.params` も dict なので、書いたキーだけが変わります。

```yaml
base: fe_hx_pressure_lags.yaml
name: fe_hx_pressure_lags_lr002
model:
  params: {learning_rate: 0.02}   # n_estimators などは土台のまま
```

### モデルを替える（共通設定とモデル設定を分ける）

`model.name` だけを書き換えると、土台の `model.params`（LightGBM の `num_leaves` など）が**残ってしまいます**（[制約](#制約と注意点)）。
モデルを比べるときは、`hx_lgbm.yaml` のようにモデルの設定を別のファイルにして、`base` を複数並べます。

```yaml
# configs/experiments/common/xgb.yaml（例）
model: {name: xgboost, params: {n_estimators: 3000, learning_rate: 0.05}, early_stopping_rounds: 100}

# configs/experiments/hx_xgb.yaml（例）
base: [common/hx_dummy.yaml, common/xgb.yaml]
name: hx_xgb
```

### 再帰予測の設定を外す

`null` を書くと None になります（None を許す項目だけ）。

```yaml
base: ts_base.yaml
name: ts_no_forecast
forecast: null
```

### ブロックの中でブロックを使う

```yaml
# configs/features/hx_all_steps.yaml（例）
description: 薬剤種類のエンコーディング＋圧力損失のラグ
steps:
  - use: hx_chemical_ordinal
  - use: hx_pressure_drop_lags
```

### 実験と同じ場所に置いたブロックを使う

`.yaml` で終わる値は、`use` を書いたファイルからの相対パスです。

```yaml
features:
  - use: blocks/my_trial.yaml   # configs/experiments/blocks/my_trial.yaml
```

### アンサンブルの設定で使う

`base` はアンサンブルの設定（`load_ensemble_config`）でも使えます。

```yaml
base: blend_common.yaml
name: blend_weighted
method: weighted
```

---

# 設計仕様

## 展開の手順

`modeling.config.load_config_dict(path, feature_blocks_dir=None)` が、1つの YAML ファイルを次の順で処理します。
`load_experiment_config` / `load_ensemble_config` は、この結果を pydantic で検証します。

1. **読み込み:** YAML を読みます。空のファイルは `{}` として扱い、最上位が mapping でなければエラーにします。
2. **`use` の展開:** `features` がリストなら、その中の `{use: ...}` をブロックの `steps` に展開します。
   - 展開は**そのファイルの中で**行います。そのため、相対パスの基準は `use` を書いたファイルになります。
3. **`base` の取り出し:** `base` を取り出します（展開結果には残しません）。
4. **土台の解決:** `base` の各ファイルについて、1〜4 を再帰的に行います。
5. **マージ:** `base` に書いた順に土台を重ね、最後に自分のキーを重ねます。

```
child.yaml ─(use展開)─┐
                      ├─ deep_merge(deep_merge({}, base1), base2) に child を重ねる
base1.yaml ─(use展開・その base も再帰)─┘
base2.yaml ─(同上)
```

## `base` の仕様

| 項目 | 仕様 |
|---|---|
| 値 | YAML のパス（文字列）、またはそのリスト |
| パスの基準 | `base` を書いたファイルのディレクトリ。絶対パスも可 |
| 複数指定 | リストの前から順に重ねる（後ろが優先）。自分のキーが最優先 |
| 多段 | 土台の YAML も `base` を持てる |
| 循環 | 自分自身を経由する `base` はエラー |

## マージの規則

| 土台の値 | 上書きの値 | 結果 |
|---|---|---|
| dict | dict | キーごとに再帰的にマージ |
| 何でも | dict 以外（リスト・文字列・数値・真偽値） | 上書きの値で**置き換え** |
| 何でも | `null` | None（その項目が None を許さなければ検証エラー） |
| dict 以外 | dict | 上書きの値で置き換え |
| なし | 何でも | 追加 |

キーを消して既定値に戻す書き方はありません。既定値に戻したいときは、値を明示してください（例: `explain: {enabled: true}`）。

## `use`（特徴量ブロック）の仕様

| 項目 | 仕様 |
|---|---|
| 書ける場所 | `features` のリストの要素（ブロックの `steps` の中も可） |
| 要素の形 | `{use: <文字列>}` のみ。ほかのキー（`params` など）を付けるとエラー |
| 名前の解決 | `.yaml` / `.yml` で終わる値は、`use` を書いたファイルからの相対パス。それ以外は `<ブロックの置き場所>/<値>.yaml` |
| ブロックの置き場所 | 既定は `configs/features/`（`load_experiment_config(path, feature_blocks_dir=...)` で変更可） |
| ブロックファイル | `steps`（リスト）が必須。`description` など、ほかのキーは読まない（説明用） |
| 展開の順序 | `use` の位置に `steps` を同じ順で差し込む |
| 循環 | ブロックが自分自身を経由して `use` されるとエラー |

## エラー

いずれも `ValueError` で、原因のファイルがメッセージに入ります。設定の値の誤り（型・未知のキーなど）は、従来どおり展開後に pydantic の `ValidationError` になります。

| 状況 | メッセージの例 |
|---|---|
| `base` / ブロックのファイルが無い | `設定ファイルが見つかりません: ...` / `特徴量ブロックが見つかりません: <名前>（<パス>）` |
| `base` が循環 | `base が循環しています: a.yaml → b.yaml → a.yaml` |
| ブロックが循環 | `特徴量ブロックが循環しています: x.yaml → y.yaml → x.yaml` |
| `base` の値が文字列・リストでない | `base にはYAMLのパス（またはそのリスト）を書いてください` |
| `use` の要素にほかのキーがある | `use は - use: <ブロック名> の形で、他のキーを付けずに書いてください` |
| ブロックに `steps` が無い | `特徴量ブロックには steps（ステップのリスト）が必要です` |
| YAML の最上位が mapping でない | `YAMLの最上位はキーと値の組（mapping）にしてください` |

## 記録と再現性

- 展開済みの設定には `base` も `use` も残りません。出力の `config.yaml` と MLflow のパラメータには、実際に使った設定が記録されます。
- そのため、あとで土台やブロックを書き換えても、過去の実行の内容は出力から分かります。
- どの土台・ブロックから作られたかは記録されません。必要なら `name` に表すか、元の YAML を git で管理してください。

## 後方互換

`base` / `use` を使わない YAML は、従来とまったく同じに読み込まれます。

## 制約と注意点

| 注意点 | 内容 |
|---|---|
| リストは置き換え | `features` に1つ足したいときも、土台の要素を全部書く（追記の書き方はない） |
| モデルの切り替え | `model.params` は dict なのでマージされる。`model.name` だけ替えると、土台のモデル固有のパラメータが残る。モデルの設定は別ファイルに分けて `base` を並べる（[レシピ](#モデルを替える共通設定とモデル設定を分ける)） |
| `name` | 書き忘れると土台と同じ名前になり、出力ディレクトリと MLflow の run 名が区別しにくくなる |
| CV・seed | 比較するパターン同士で `cv` と `seed` を変えない（変えるとスコアの差が特徴量の差でなくなる） |
| ブロックの引数 | `use` にパラメータは渡せない。値を変えたいときは別のブロックを作るか、ステップを直接書く |
| 置けない場所 | `use` は `features` の中だけ。`base` はファイルの最上位だけ |

## 実装

| 対象 | 場所 |
|---|---|
| 展開 | `src/modeling/config.py` の `load_config_dict`。内部で `_resolve_config`（`base` の再帰）・`_expand_feature_blocks`（`use` の再帰）・`_deep_merge`（マージ）を使う |
| ブロックの既定の置き場所 | `default_feature_blocks_dir()`（`configs/features/`） |
| テスト | `tests/test_modeling_config.py`（継承・ブロック・エラー・同梱の雛形の検証） |
