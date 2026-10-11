# configs

実験・アンサンブル・ハイパーパラメータ探索の設定（YAML）を置くディレクトリです。
設定はすべて pydantic で検証されます。未知のキー・型の誤り・矛盾した組み合わせは、学習を始める前にエラーになります。

| ディレクトリ | 内容 | 使うスクリプト |
|---|---|---|
| [experiments/](experiments/README.md) | 1つのモデルの実験設定（データ・特徴量・CV・モデル・指標） | `scripts/run_experiment.py` |
| [ensembles/](ensembles/README.md) | 実行済みの実験の予測を組み合わせるアンサンブル設定 | `scripts/run_ensemble.py` |
| [pipelines/](pipelines/README.md) | 複数モデルの探索 → 学習 → アンサンブルを一括で行う設定 | `scripts/run_pipeline.py` |
| [features/](features/README.md) | 実験設定から `use:` で呼び出す特徴量ブロック | （実験設定から参照） |
| [search_spaces/](search_spaces/README.md) | モデルごとの Optuna の探索範囲 | （実験設定から `base:` で継承） |

## 設定どうしの関係

```
pipelines/<名前>.yaml ──members──▶ 実験設定 ──base──▶ 共通設定・search_spaces/<モデル>.yaml
                                      └──use──▶ features/<ブロック>.yaml
experiments/<名前>.yaml ──base──▶ 共通の実験設定（例: fe_base.yaml）
ensembles/<名前>.yaml ──members──▶ 実行済みの実験の出力（outputs/experiments/）
```

## 共通の書き方

- **`base: <YAML>`**（またはリスト）: 別の設定を土台にして、書いたキーだけを上書きします。
  - dict は再帰的にマージし、リストは置き換えます。
  - パスは、`base` を書いたファイルからの相対パスです。
- **`features: [{use: <ブロック名>}]`**: `features/<ブロック名>.yaml` のステップに展開します。
- **展開結果の確認:** 次のコマンドで確認できます。

```bash
uv run python -c "import sys, yaml; sys.path.insert(0, 'src'); from pathlib import Path; from modeling.config import load_config_dict; print(yaml.safe_dump(load_config_dict(Path('configs/experiments/fe_date_parts.yaml')), allow_unicode=True, sort_keys=False))"
```

## 関連ドキュメント

- [docs/modules/modeling.md](../docs/modules/modeling.md): 実験・アンサンブル設定の全項目
- [docs/guides/feature_exploration.md](../docs/guides/feature_exploration.md): `base` / `use` の仕様と、特徴量の探索の進め方
- [docs/guides/hyperparameter_tuning.md](../docs/guides/hyperparameter_tuning.md): 探索範囲・パイプラインの仕様と使い方

## 注意

- 同梱の例の目的変数 `合計` は仮置きです。コンペの目的変数が決まったら、各設定の `data.target` と `drop_cols` を見直してください。
- データファイル（`data/`）は git 管理外です。設定は、各自の環境にあるファイルのパスを指します。
