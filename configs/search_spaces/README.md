# search_spaces

モデルごとの Optuna の探索範囲です。実験設定から `base` で継承して使います。
各ファイルには次の2つを書きます。

- **`model`:** 探索しない固定値（モデル名・`params`・`early_stopping_rounds`）。
- **`tuning.search_space`:** 探索範囲。

```yaml
# 実験設定（例: configs/pipelines/example_ensemble/lgbm.yaml）
base: [common.yaml, ../../search_spaces/lightgbm.yaml]
name: example_ensemble_lgbm
features:
  - use: categorical_ordinal
```

## ファイル

| ファイル | モデル | 探索するパラメータ |
|---|---|---|
| `lightgbm.yaml` | LightGBM（`n_estimators: 5000` ＋ early stopping 100） | `learning_rate`・`num_leaves`・`min_child_samples`・`subsample`・`colsample_bytree`・`reg_alpha`・`reg_lambda`（`subsample_freq` は 1 に固定） |
| `xgboost.yaml` | XGBoost（`n_estimators: 5000` ＋ early stopping 100） | `learning_rate`・`max_depth`・`min_child_weight`・`subsample`・`colsample_bytree`・`reg_alpha`・`reg_lambda` |
| `linear.yaml` | 線形モデル（回帰: Ridge） | `alpha`（分類で使う場合は `C` に書き換える。ファイル内のコメント参照） |

## 書き方

```yaml
model:
  name: lightgbm
  params: {n_estimators: 5000}      # 探索しない固定値
  early_stopping_rounds: 100
tuning:
  search_space:
    learning_rate: {type: float, low: 0.005, high: 0.2, log: true}
    num_leaves: {type: int, low: 8, high: 256, log: true}
    boosting_type: {type: categorical, choices: [gbdt, dart]}
    subsample_freq: {type: fixed, value: 1}
```

| `type` | 項目 | 内容 |
|---|---|---|
| `float` | `low`, `high`, `log`, `step` | 実数（`log: true` で対数スケール） |
| `int` | `low`, `high`, `log`, `step` | 整数 |
| `categorical` | `choices` | 候補から選ぶ |
| `fixed` | `value` | 探索せず固定 |

## 使い方のポイント

- **範囲を一部だけ変える:** `search_space` は dict なので、継承した実験設定で、パラメータ単位に範囲を上書きできます。

  ```yaml
  base: [common.yaml, ../../search_spaces/lightgbm.yaml]
  tuning:
    search_space:
      learning_rate: {type: float, low: 0.005, high: 0.05, log: true}   # この項目だけ変える
      num_leaves: {type: fixed, value: 31}                              # 探索をやめて固定する
  ```

- **既定と置き換え:**
  - `search_space` を書くと、モデルの既定の探索範囲を**置き換え**ます。
  - 探索したパラメータは、`model.params` の固定値より優先されます。
- **木の本数:** 探索せず、大きめの `n_estimators` と `early_stopping_rounds` で決めます。試行ごとに検証 fold で打ち切られます。
- **study の切り替え:** 探索範囲を変えると、その実験の Optuna の study は自動で新しくなります（条件の違う試行が混ざりません）。
- **モデルの切り替え:** `model.params` もマージされるので、`model.name` だけを書き換えると、別モデルのパラメータが残ります。モデルごとにファイルを分け、`base` で選んでください。

詳しい仕様は [docs/guides/hyperparameter_tuning.md](../../docs/guides/hyperparameter_tuning.md) を参照してください。
