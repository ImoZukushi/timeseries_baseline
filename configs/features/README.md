# 特徴量ブロック

実験設定（`configs/experiments/*.yaml`）の `features` から、名前で呼び出せる特徴量ステップのまとまりです。
同じ組み合わせを複数の実験で使い回し、パターンを差し替えて比較するために使います。

## 書き方

ブロックのファイルには `steps`（`features` と同じ書式のリスト）を書きます。`description` は任意です。

```yaml
# configs/features/categorical_ordinal.yaml
description: 列車番号・停車駅名の序数エンコーディング
steps:
  - class: feature_engineering.categorical.PolarsOrdinalEncoder
    params: {variables: [列車番号, 停車駅名]}
```

実験設定では、`features` の要素に `use` で書きます。展開されたステップは、並んだ順に Pipeline に入ります。

```yaml
features:
  - use: categorical_ordinal            # configs/features/categorical_ordinal.yaml
  - use: date_parts
  - use: ../features/my_block.yaml      # .yaml で終わる値は、この YAML からの相対パス
  - class: modeling.pipeline.DropColumns  # 通常のステップと混ぜてよい
    params: {columns: [不要な列]}
```

- ブロックの中でも `use` を使えます（循環するとエラーになります）。
- `use` の要素には、ほかのキー（`params` など）を付けられません。パラメータを変えたいときは、別のブロックを作ります。

## 一覧

| ブロック | 内容 |
|---|---|
| `categorical_ordinal` | 列車番号・停車駅名の序数エンコーディング |
| `categorical_target` | 出現の少ない列車番号を `Other` にまとめてから、列車番号・停車駅名をターゲットエンコーディング |
| `date_parts` | 年月日から月・日・曜日を作り、年月日を削除する（使う実験では `data.drop_cols` から `年月日` を外す） |

使える transformer は [docs/modules/feature_engineering.md](../../docs/modules/feature_engineering.md) を参照してください。
探索の進め方・レシピ・詳しい仕様は [docs/guides/feature_exploration.md](../../docs/guides/feature_exploration.md) にあります。
