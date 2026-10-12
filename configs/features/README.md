# 特徴量ブロック

実験設定（`configs/experiments/*.yaml`）の `features` から、名前で呼び出せる特徴量ステップのまとまりです。
同じ組み合わせを複数の実験で使い回し、パターンを差し替えて比較するために使います。

## 書き方

ブロックのファイルには `steps`（`features` と同じ書式のリスト）を書きます。`description` は任意です。

```yaml
# configs/features/hx_chemical_ordinal.yaml
description: 薬剤種類の序数エンコーディング（熱交換器ダミーデータ）
steps:
  - class: feature_engineering.categorical.PolarsOrdinalEncoder
    params: {variables: [薬剤種類]}
```

実験設定では、`features` の要素に `use` で書きます。展開されたステップは、並んだ順に Pipeline に入ります。

```yaml
features:
  - use: hx_chemical_ordinal            # configs/features/hx_chemical_ordinal.yaml
  - use: hx_pressure_drop_lags
  - use: ../features/my_block.yaml      # .yaml で終わる値は、この YAML からの相対パス
  - class: modeling.pipeline.DropColumns  # 通常のステップと混ぜてよい
    params: {columns: [不要な列]}
```

- ブロックの中でも `use` を使えます（循環するとエラーになります）。
- `use` の要素には、ほかのキー（`params` など）を付けられません。パラメータを変えたいときは、別のブロックを作ります。

## 一覧

| ブロック | 内容 |
|---|---|
| `hx_chemical_ordinal` | 熱交換器ダミーデータ: 薬剤種類の序数エンコーディング（`configs/experiments/common/hx_dummy.yaml`・特徴量の探索のグループ `chemical_type` で使用） |
| `hx_pressure_drop_lags` | 熱交換器ダミーデータ: 圧力損失のラグ（1・7・14行前）と30行の移動平均（機器番号ごと。`fe_hx_pressure_lags`・グループ `pressure_lags` で使用） |

ほかの設定ディレクトリとの関係は [configs/README.md](../README.md) を参照してください。
使える transformer は [docs/modules/feature_engineering.md](../../docs/modules/feature_engineering.md) を参照してください。
探索の進め方・レシピ・詳しい仕様は [docs/guides/feature_exploration.md](../../docs/guides/feature_exploration.md) にあります。
