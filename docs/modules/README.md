# `src/` モジュールリファレンス

`src/` 以下の再利用可能なモジュールの仕様と使い方をまとめたドキュメントです。
スクリプト・ノートブック・アプリから `import` して使う部品を、パッケージごとに説明します。

| パッケージ | 役割 | ドキュメント |
|---|---|---|
| `util` | パス・CSV読み込み（エンコーディング自動判定）・matplotlibの日本語設定 | [util.md](util.md) |
| `eda` | CSVのデータ品質チェック・時系列EDA（変換系列・移動平均・ACF/PACF） | [eda.md](eda.md) |
| `feature_engineering` | polars DataFrame を受け取る sklearn 互換の特徴量 transformer | [feature_engineering.md](feature_engineering.md) |
| `modeling` | 実験の実行基盤（設定・CV・モデル・再帰予測・チューニング・SHAP・アンサンブル・実験ログ） | [modeling.md](modeling.md) |
| `evaluation` | 誤差評価の可視化（残差・時系列の残差診断・分類・学習曲線・影響度・SHAPの相関） | [evaluation.md](evaluation.md) |

## 依存関係

```
util  ←  eda  ←  evaluation  ←  modeling
  ↑                                ↑
  └──── feature_engineering ───────┘
```

- `util` はどこからでも使う最下層です。
- `feature_engineering` は `util` 以外に依存せず、`modeling` の Pipeline と `eda` の変換系列から使われます。
- `evaluation` は予測値・実測値・SHAP値などの配列だけを受け取り、`modeling` に依存しません。
  ノートブックで単独で使えます。
- `modeling` は上のすべてを組み合わせて1実験を実行します。

## 共通の約束

| 項目 | 方針 |
|---|---|
| DataFrame | polars が基本です。pandas はモデルの入力（`modeling.pipeline.ToModelInput`）と seaborn など、ライブラリが要求する箇所だけで使います |
| transformer | `feature_engineering` のクラスはすべて sklearn の `fit` / `transform` 規約に従います。`sklearn.pipeline.Pipeline` に入れられます |
| リーク防止 | `fit` は学習データだけに呼びます。`modeling` の CV は fold ごとに Pipeline を作り直し、学習 fold だけで fit します |
| 図 | 関数の中で `util.plotting.ensure_japanese_font()` を呼ぶので、日本語の見出しはそのまま表示されます。保存先は `outputs/figures/` などを呼び出し側が渡します |
| 生データ | `data/raw`・`data/external` は読むだけで、書き込みません。派生データは `data/interim`・`data/processed`・`outputs/` に置きます |
| パス | `util.paths` の `data_dir()`・`outputs_dir()` を起点に組み立てます（絶対パスを書かない） |

## import のしかた

`src/` を import パスに入れて、パッケージ名から import します（`src.` は付けません）。

- `pytest` / `mypy`: `pyproject.toml` で `src` を設定済みです。
- スクリプト: `scripts/*.py` は先頭で `src/` を `sys.path` に加えています。
- ノートブック: `notebook/*.ipynb` の最初のセルで同様に加えています。

```python
from util.csv_io import read_csv_auto
from modeling.config import ExperimentConfig
```

## やりたいこと別の入口

| やりたいこと | 使うもの |
|---|---|
| CP932 の CSV を読みたい・日時列を自動で日時型にしたい | `util.csv_io.read_csv_auto` |
| 数百MBの CSV をメモリに載せずに処理したい | `util.csv_io.scan_csv_auto` |
| CSV の欠損・分布・相関をまとめて確認したい | `eda.csv_quality.run_quality_checks`（CLI: `scripts/run_csv_quality_checks.py`） |
| 時系列の定常性・自己相関を見たい | `eda.time_series_eda.run_time_series_checks` |
| ラグ・移動平均・日付の特徴量を作りたい | `feature_engineering.time_series` / `datetime_features` |
| 予測値を元の尺度に戻せる差分・対数変換をしたい | `feature_engineering.series_transform` |
| YAML 1つで CV学習・予測・SHAP・誤差評価・MLflow記録まで回したい | `modeling.experiment.run_experiment`（CLI: `scripts/run_experiment.py`） |
| 時系列を数期先まで予測したい（予測値をラグに使う） | `modeling` の `forecast` 設定（再帰予測） |
| 複数モデルの予測をアンサンブルしたい | `modeling.ensemble.run_ensemble`（CLI: `scripts/run_ensemble.py`） |
| 残差・混同行列・ROC などを描きたい | `evaluation` の各 `XxxDisplay` |

## 使い方の実例

| 実例 | 内容 |
|---|---|
| `scripts/quickstart_delhi_climate.py` / `notebook/001_quickstart_delhi_climate.ipynb` | 再帰予測・Optuna・SHAP・誤差評価・アンサンブルの一連の流れ |
| `scripts/eda_delhi_climate.py` / `notebook/002_eda_delhi_climate.ipynb` | `eda` の使い方 |
| `configs/experiments/*.yaml` / `configs/ensembles/*.yaml` | 実験・アンサンブルの設定の例 |
| `tests/test_*.py` | 各関数の細かな挙動（境界条件・入力の形） |

ディレクトリ全体の構成は [../agent/repository-structure.md](../agent/repository-structure.md) を参照してください。
