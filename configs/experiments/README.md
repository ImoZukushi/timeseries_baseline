# experiments

1つのモデルの実験設定です。データ・特徴量・CV・モデル・評価指標を1つの YAML に書き、`scripts/run_experiment.py` で実行します。
CV学習 → 予測の保存 → SHAP → 誤差評価 → MLflow への記録までを、まとめて行います。

```bash
uv run python scripts/run_experiment.py --config configs/experiments/example_lgbm.yaml
uv run python scripts/run_experiment.py --config "configs/experiments/fe_*.yaml"      # 複数（glob）
uv run python scripts/run_experiment.py --config configs/experiments/example_lgbm.yaml --tune --n-trials 30
```

| オプション | 内容 |
|---|---|
| `--tune` | Optuna で探索してから学習する |
| `--n-trials` | study 全体での試行回数の目標 |
| `--no-explain` | SHAP を省く |
| `--no-tracking` | MLflow に記録しない |
| `--output-root` | 出力のルート（既定 `outputs/experiments/`） |

出力は `outputs/experiments/<name>/<日時>/` です。主なファイル:

| ファイル | 内容 |
|---|---|
| `config.yaml` | 展開済みの設定 |
| `cv_scores.csv` | fold ごとのスコア |
| `oof_predictions.parquet` / `test_predictions.parquet` | OOF 予測・テスト予測 |
| `shap/` | SHAP の結果 |
| `evaluation/` | 誤差評価の結果 |

## ファイル

| ファイル | 内容 |
|---|---|
| `example_lgbm.yaml` | LightGBM 回帰の動作確認用（group CV、early stopping） |
| `example_dummy.yaml` | 定数予測のベースライン（`example_lgbm` と比べて、最低限超えるべき水準を確認する） |
| `fe_base.yaml` | 特徴量の探索の共通設定とベースライン（SHAP・誤差評価は無効、MLflow の実験名 `feature_search`） |
| `fe_target_encoding.yaml` | `base: fe_base.yaml` ＋ 低頻度まとめ＋ターゲットエンコーディング |
| `fe_date_parts.yaml` | `base: fe_base.yaml` ＋ 日付の成分（`data.drop_cols` だけを上書きする例を兼ねる） |

`example_lgbm` / `example_dummy` は、`configs/ensembles/example_blend.yaml` の構成要素です。

## 最小の例

```yaml
name: my_experiment          # 出力ディレクトリ名・MLflow の run 名（実験ごとに変える）
task: regression             # binary / multiclass / regression / time_series
data:
  train_path: data/processed/train_with_departure_time.csv
  target: 合計
  group_col: 年月日
  drop_cols: [年月日, 発車時刻]
features:
  - use: categorical_ordinal # configs/features/categorical_ordinal.yaml
cv: {method: group, n_splits: 5, seed: 42}
model:
  name: lightgbm
  params: {n_estimators: 2000, learning_rate: 0.05}
  early_stopping_rounds: 100
metrics: [rmse, mae]         # 先頭が主指標
```

差分だけを書く場合（特徴量の探索など）は、`base` を使います。

```yaml
base: fe_base.yaml
name: fe_my_trial
features:
  - use: categorical_ordinal
  - use: date_parts
```

## 書くときの注意

- **`name`:** パターンごとに変えます。同じ名前だと出力ディレクトリ・MLflow の run・Optuna の study が混ざって見づらくなります。
- **比べるときの CV:** 比べる実験どうしで `cv`（と `seed`）を変えないでください。スコアの差が、比べたいものの差でなくなります。
- **リーク:** 目的変数の内訳など、予測時に使えない列は `drop_cols` で除きます。例の `フェンダー部分(…)`・`台車部分` は `合計` の内訳です。
- **部品の置き場所:** このディレクトリには、単独で実行できる設定だけを置きます。`base` 専用の部品（`name` や `model` の無いファイル）を置くと、`--config "configs/experiments/*.yaml"` で検証エラーになります。部品は `configs/search_spaces/` やパイプラインの下に置いてください。

全項目は [docs/modules/modeling.md](../../docs/modules/modeling.md#設定リファレンスexperimentconfig) を参照してください。
特徴量の探索の進め方は [docs/guides/feature_exploration.md](../../docs/guides/feature_exploration.md) にあります。
