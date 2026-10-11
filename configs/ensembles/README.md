# ensembles

実行済みの実験の予測（OOF 予測・テスト予測）を組み合わせるアンサンブルの設定です。
モデルは再学習せず、各実験が保存した予測ファイルを読み込んで統合します。

```bash
# 構成要素の実験を先に実行しておく
uv run python scripts/run_experiment.py --config configs/experiments/example_lgbm.yaml configs/experiments/example_dummy.yaml
uv run python scripts/run_ensemble.py --config configs/ensembles/example_blend.yaml
```

出力は `outputs/ensembles/<name>/<日時>/` です。

| ファイル | 内容 |
|---|---|
| `ensemble_scores.csv` | 構成要素とアンサンブルの OOF スコア |
| `weights.csv` | 構成要素の重み |
| `oof_predictions.parquet` / `test_predictions.parquet` | アンサンブルの予測 |

探索から学習・アンサンブルまでを一括で行う場合は、[configs/pipelines/](../pipelines/README.md) を使ってください。
複数の統合方法をまとめて比べられ、構成要素も「その実行の結果」に固定されます。

## ファイル

| ファイル | 内容 |
|---|---|
| `example_blend.yaml` | `example_lgbm` と `example_dummy` の加重平均（`weighted`） |

## 書き方

```yaml
name: my_blend
task: regression
members:                                       # 2つ以上
  - {name: lgbm, experiment: example_lgbm}     # outputs/experiments/example_lgbm/ の最新の実行
  - {name: xgb, path: outputs/experiments/x/20260101_000000_000000}  # 出力ディレクトリを直接指定（リポジトリルート基準）
  - {name: nn, run_id: 0123abcd...}            # MLflow の run の artifact から読む
method: weighted      # mean / rank_mean（二値分類のみ）/ weighted / stacking
metrics: [rmse, mae]  # 先頭が重み・メタモデルの最適化と比較の基準
stacking: {model: {name: linear}}              # method: stacking のメタモデル
```

| `method` | 内容 |
|---|---|
| `mean` | 単純平均 |
| `rank_mean` | 順位の平均（二値分類・AUC 向け） |
| `weighted` | OOF で主指標が最良になる重み（非負・合計1） |
| `stacking` | 構成要素の予測を特徴量にしたメタモデル |

`weighted` / `stacking` のスコアは、「fold k 以外で学習して fold k を予測する」cross-fitting で計算します（同じ OOF で学習・評価すると楽観的になるため）。

## 注意

- **同じデータ・CV分割:** 構成要素は、同じ学習データ・同じ CV 分割（`cv` と `seed`）で作られている必要があります。行数・fold・目的変数が一致しなければエラーになります。
- **`experiment:` 指定:** 実行のたびに「最新の実行」を探します。古い結果を使い続けたいときは、`path:` で出力ディレクトリを固定します。
- **`base`:** 使えます（例: 共通の `members` を土台にして `method` だけ変える）。

全項目は [docs/modules/modeling.md](../../docs/modules/modeling.md#アンサンブルensembleconfig) を参照してください。
