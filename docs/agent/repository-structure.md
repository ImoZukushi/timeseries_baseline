# リポジトリ構成

各ディレクトリの役割を説明します。

| ディレクトリ | 役割 |
|-------------|------|
| `src/` | 再利用可能なPythonモジュール（下表） |
| `configs/experiments/` | モデル学習実験の設定YAML（`scripts/run_experiment.py` 用） |
| `configs/ensembles/` | アンサンブルの設定YAML（`scripts/run_ensemble.py` 用） |
| `notebook/` | 探索・分析用Jupyter Notebook |
| `scripts/` | 実行スクリプト |
| `tests/` | pytest用テスト |
| `model/` | 学習済みモデル |
| `app/` | アプリケーション |
| `data/raw/` | 元データ（不変・gitignore対象） |
| `data/external/` | 外部データ（不変・gitignore対象） |
| `data/interim/` | 中間加工データ（gitignore対象） |
| `data/processed/` | 最終加工データ（gitignore対象） |
| `outputs/` | グラフ・集計テーブル・レポート |
| `outputs/experiments/`, `outputs/ensembles/` | 実験・アンサンブルの予測とスコア（gitignore対象） |
| `outputs/optuna/` | Optuna studyのSQLite（gitignore対象） |
| `mlruns/` | ローカルMLflowの実験ログ（gitignore対象） |
| `docs/agent/` | エージェント向けプロジェクト文書 |
| `.claude/skills/` | 作業別スキルファイル（Claude Code） |

## `src/` 配下のパッケージ

| パッケージ | 役割 |
|-----------|------|
| `src/util/` | パス・CSV読み込み・matplotlib設定などの汎用ヘルパー |
| `src/eda/` | データ品質チェック・時系列EDA |
| `src/feature_engineering/` | sklearn互換の特徴量エンジニアリングtransformer |
| `src/modeling/` | モデル学習基盤（CV・モデル・チューニング・SHAP・実験ログ・アンサンブル） |

## モデリングの実行手順

```bash
# 実験（CV学習・予測・SHAP・MLflow記録）
uv run python scripts/run_experiment.py --config "configs/experiments/*.yaml"
# ハイパーパラメータチューニング付き
uv run python scripts/run_experiment.py --config configs/experiments/example_lgbm.yaml --tune --n-trials 50
# アンサンブル
uv run python scripts/run_ensemble.py --config configs/ensembles/example_blend.yaml
# 実験ログの閲覧
uv run mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# NN（mlp / cnn1d）を使う場合は任意依存を入れる
uv sync --extra nn
```

新しいモデルは `src/modeling/models/` に `ModelSpec` を継承したクラスを追加し
`@register_model` を付けると、YAMLの `model.name` で指定できるようになる。
