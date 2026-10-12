# pipelines

複数のモデルについて「Optuna で探索 → 最良パラメータで学習 → 複数の方法でアンサンブル → 比較」を一括で行う設定です。
`scripts/run_pipeline.py` で実行します。

```bash
uv run python scripts/run_pipeline.py --config configs/pipelines/hx_ensemble.yaml --check   # 設定の検証のみ
uv run python scripts/run_pipeline.py --config configs/pipelines/hx_ensemble.yaml
```

| オプション | 内容 |
|---|---|
| `--check` | 設定の検証だけを行う（学習しない） |
| `--skip-tuning` | 探索せずに学習する |
| `--no-tracking` | MLflow に記録しない |
| `--output-root` | 出力のルート（既定 `outputs/pipelines/`） |
| `--optuna-dir` | study の保存先（既定 `outputs/optuna/`） |

出力は `outputs/pipelines/<name>/<日時>/` です。

| ファイル | 内容 |
|---|---|
| `summary.csv` | 単体モデルと各アンサンブルの OOF スコア・順位 |
| `final.yaml` | 最良のもの（最終モデル）と、テスト予測のパス |
| `members.csv` | 各モデルの study・試行数・CV スコア |
| `best_params.yaml` | 各モデルの、学習に使ったパラメータ |

## 構成

パイプライン設定（`<名前>.yaml`）と、その構成モデルの実験設定（`<名前>/`）を並べて置きます。

```
pipelines/
├── hx_ensemble.yaml       # パイプライン設定
└── hx_ensemble/           # 構成モデルの実験設定
    ├── common.yaml             # 共通設定（base: ../../experiments/common/hx_dummy.yaml ＋ 探索向けの設定）。全構成モデルが base で継承
    ├── lgbm.yaml               # base: [common.yaml, ../../search_spaces/lightgbm.yaml]
    ├── xgb.yaml                # base: [common.yaml, ../../search_spaces/xgboost.yaml]
    └── linear.yaml             # base: [common.yaml, ../../search_spaces/linear.yaml]
```

構成モデルの設定は、`configs/experiments/` ではなくここに置きます。
`common.yaml` のような部品が、`run_experiment.py --config "configs/experiments/*.yaml"` に混ざらないようにするためです。
構成モデルの設定は普通の実験設定なので、1つだけなら `run_experiment.py` でも実行できます。

## 書き方

```yaml
name: hx_ensemble
members:                            # パスはこの YAML からの相対パス
  - config: hx_ensemble/lgbm.yaml
    n_trials: 30                    # study 全体での試行回数の目標（足りない分だけ実行）
    timeout: 1800                   # 今回の実行での打ち切り秒数（任意）
  - config: hx_ensemble/xgb.yaml
    n_trials: 30
  - config: hx_ensemble/linear.yaml
    tune: false                     # 探索せず、実験設定の params で学習する
ensemble:
  methods: [mean, weighted, stacking]   # すべて実行して比べる
  metrics: null                         # 省略時は構成モデルの metrics
  stacking: {model: {name: linear}}
tracking: {experiment_name: hx_ensemble}
```

## 動作の要点

- **実行前の検証:** 全構成モデルが、同じデータ・同じ CV 分割・同じ主指標かを確認します（アンサンブルの前提）。
  特徴量・モデル・探索範囲は、構成モデルごとに変えてかまいません。
- **`n_trials` は合計の目標:**
  - 再実行すると、足りない分だけ探索します。
  - 目標に達したモデルは探索を飛ばして学習だけを行います。
  - 中断しても、同じコマンドで続きから進みます。
- **study の切り替え:** 探索の条件（データの内容・特徴量・CV・モデル・探索範囲など）を変えると、自動で新しい study になります。
- **アンサンブルの構成要素:** その実行で学習した結果を使います。

## 新しいパイプラインを作る

1. `hx_ensemble.yaml` と `hx_ensemble/` をコピーし、名前を変える。
2. `common.yaml` の継承元（データ・特徴量・CV・指標）を確認する。熱交換器のデータなら `experiments/common/hx_dummy.yaml` のままでよい。
3. 構成モデルの `name` と `features` を決める。探索範囲は `configs/search_spaces/` から継承し、必要な項目だけ上書きする。
4. `--check` で検証してから実行する。

動作確認だけを速く回すときは、`base` で継承して試行回数だけを小さくした設定を作ると便利です。

```yaml
base: hx_ensemble.yaml
name: hx_ensemble_small
members:          # リストは置き換えなので全構成モデルを書く（パスは土台のファイル基準のまま使える）
  - {config: hx_ensemble/lgbm.yaml, n_trials: 3}
  - {config: hx_ensemble/xgb.yaml, n_trials: 3}
  - {config: hx_ensemble/linear.yaml, n_trials: 3}
```

詳しい仕様は [docs/guides/hyperparameter_tuning.md](../../docs/guides/hyperparameter_tuning.md) を参照してください。
