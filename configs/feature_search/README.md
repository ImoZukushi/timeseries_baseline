# 特徴量の探索の設定

単一モデルで特徴量の組み合わせを比べる探索の設定です。`scripts/run_feature_search.py` で実行します。

```bash
uv run python scripts/run_feature_search.py --config configs/feature_search/<名前>.yaml
uv run python scripts/run_feature_search.py --config configs/feature_search/<名前>.yaml --check   # 確認のみ
```

特徴量を**グループ**（一緒に足し引きする列）に分け、グループの組み合わせを**パターン**として並べます。
パターンごとに同じ CV 分割で学習し、基準のパターンとの差を比べます。
あるパターンからグループを1つずつ外す**アブレーション**も自動で行えます。

書き方・出力・仕様は [docs/guides/feature_search.md](../../docs/guides/feature_search.md) を参照してください。

## ファイル

| ファイル | 内容 |
|---|---|
| `hx_dummy.yaml` | 熱交換器ダミーデータ（前処理済み、`data/processed/dummy/`）を LightGBM で比べる探索。元データの列だけのベースラインに前処理の特徴量グループを順に足すパターンと、全部入りからのアブレーション |
| `hx_dummy/experiment.yaml` | `hx_dummy.yaml` の全パターン共通の実験設定。データ・年ごとの時系列 CV・指標・LightGBM は `configs/experiments/common/` から継承し、SHAP・誤差評価を無効にする |

共通の実験設定はこのディレクトリの下（`<探索名>/experiment.yaml`）に置きます。
`configs/experiments/` に置くと、`run_experiment.py --config "configs/experiments/*.yaml"` に `name` の無い部品が混ざるためです。

## 新しい探索を作る

1. `<探索名>/experiment.yaml` に、共通の実験設定を書きます（`name` と `data.feature_cols` は書かない）。データ・CV が実験と同じなら、`configs/experiments/common/` の設定を `base` で継承します。
2. `<探索名>.yaml` に、`groups`・`patterns`・（任意で）`baseline`・`ablation` を書きます。
3. `--check` で、パターンごとのグループ・列数と、列がデータにあるかを確かめます。

グループでエンコーディングなどのステップが必要な場合は、`configs/features/` にブロックを作って `use` で呼びます。
