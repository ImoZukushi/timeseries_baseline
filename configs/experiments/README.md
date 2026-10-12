# experiments

1つのモデルの実験設定です。データ・特徴量・CV・モデル・評価指標を1つの YAML に書き、`scripts/run_experiment.py` で実行します。
CV学習 → 予測の保存 → SHAP → 誤差評価 → MLflow への記録までを、まとめて行います。

```bash
uv run python scripts/run_experiment.py --config configs/experiments/hx_lgbm.yaml
uv run python scripts/run_experiment.py --config "configs/experiments/fe_hx_*.yaml"   # 複数（glob）
uv run python scripts/run_experiment.py --config configs/experiments/hx_lgbm.yaml --tune --n-trials 30
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

熱交換器のダミーデータ（前処理済みのデータセット）で汚れ係数を予測する設定です。
先に `scripts/preprocess_heat_exchanger_dummy.py` でデータセットを作ってください。

| ファイル | 内容 |
|---|---|
| `common/hx_dummy.yaml` | 共通設定（データ・特徴量・年ごとの時系列 CV・指標）。`name` が無い部品なので、glob に入らないサブディレクトリに置く |
| `common/lgbm.yaml` | LightGBM の設定（探索しない場合の既定のパラメータ） |
| `hx_lgbm.yaml` | LightGBM 回帰。SHAP・誤差評価も作る基本の例 |
| `hx_dummy_baseline.yaml` | 定数予測のベースライン（モデルが最低限超えるべき水準） |
| `fe_hx_base.yaml` | 特徴量の探索（`base` / `use` の例）の土台（`hx_lgbm` ＋ SHAP・誤差評価なし、MLflow の実験名 `feature_exploration_hx`） |
| `fe_hx_pressure_lags.yaml` | `base: fe_hx_base.yaml` ＋ CV の中で作る圧力損失のラグ（`use` の例） |
| `fe_hx_no_flags.yaml` | `base: fe_hx_base.yaml` ＋ クリーニングの印の列を外す（`data.drop_cols` だけを上書きする例） |

`hx_lgbm` / `hx_dummy_baseline` は、`configs/ensembles/hx_blend.yaml` の構成要素です。
列のグループの組み合わせ・アブレーションを一括で比べる場合は、`configs/feature_search/` を使います。

## 最小の例

```yaml
name: my_experiment          # 出力ディレクトリ名・MLflow の run 名（実験ごとに変える）
task: regression             # binary / multiclass / regression / time_series
data:
  train_path: data/processed/dummy/train_dataset.parquet
  target: 汚れ係数
  time_col: 日付
  drop_cols: [日付, 機器]
features:
  - use: hx_chemical_ordinal # configs/features/hx_chemical_ordinal.yaml
cv:
  method: time_cutoff
  cutoffs: ["2022-01-01", "2023-01-01", "2024-01-01"]
  valid_end: "2025-01-01"
model:
  name: lightgbm
  params: {n_estimators: 3000, learning_rate: 0.05}
  early_stopping_rounds: 100
metrics: [rmse, mae]         # 先頭が主指標
```

共通設定を使い、差分だけを書く場合は `base` を使います。

```yaml
base: [common/hx_dummy.yaml, common/lgbm.yaml]
name: hx_lgbm_lr002
model:
  params: {learning_rate: 0.02}
```

## 書くときの注意

- **`name`:** パターンごとに変えます。同じ名前だと出力ディレクトリ・MLflow の run・Optuna の study が混ざって見づらくなります。
- **比べるときの CV:** 比べる実験どうしで `cv`（と `seed`）を変えないでください。スコアの差が、比べたいものの差でなくなります。
- **リーク:** 予測時に使えない列は `drop_cols` で除きます。前処理済みのデータセットは目的変数を特徴量の計算に使っていませんが、
  外れ値の判定・補間・洗浄の推定にはその日より後の特徴量を使っています（[データカタログ](../../docs/agent/data-catalog.md)）。
- **部品の置き場所:** このディレクトリの直下には、単独で実行できる設定だけを置きます。`base` 専用の部品（`name` や `model` の無いファイル）は
  `common/` などのサブディレクトリに置いてください（`--config "configs/experiments/*.yaml"` で検証エラーにならないように）。

全項目は [docs/modules/modeling.md](../../docs/modules/modeling.md#設定リファレンスexperimentconfig) を参照してください。
特徴量の探索の進め方は [docs/guides/feature_exploration.md](../../docs/guides/feature_exploration.md) にあります。
