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

一連の流れ（特徴量のPipeline化 → 再帰予測 → Optuna → SHAP → アンサンブル）を
Pythonから使う例は `scripts/quickstart_delhi_climate.py`（QuickStart）を参照。

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

### 時系列の再帰的多段予測

`task: time_series` の設定に `forecast` を書くと、目的変数のラグ等を特徴量にした1期先モデルを学習し、
検証・テスト期間は **予測値を次の時点のラグとして使いながら1ステップずつ予測** する。

```yaml
task: time_series
data: {train_path: ..., test_path: ..., target: y, time_col: date, drop_cols: [date]}
cv: {method: time_cutoff, cutoffs: ["2024-03-01", "2024-04-01"]}
forecast:
  series_col: series_id     # 複数系列の場合
  lags: [1, 2, 7]
  rolling_windows: [7]      # y_{t-1} から過去7期の平均
  horizon: 28               # バックテストで評価する最大ステップ（省略時は検証期間全体）
  clip: {min: 0}            # 再帰中の予測値の範囲制限（任意）
```

- OOF・チューニング・アンサンブルは、検証期間の実測値を使わない再帰予測のスコアで評価する。
  参考として、真のラグを使う1期先予測のスコア（`onestep_oof_*`）もMLflowに記録する。
- ステップ別の誤差は `horizon_scores.csv` / `horizon_error.png` に出力される。
- 前提: 各系列は一定間隔（1行=1ステップ）であり、`test_path` の外生変数は予測時点で既知であること。
  複数系列が混在するデータでは `cv.method: time_cutoff` を推奨。

新しいモデルは `src/modeling/models/` に `ModelSpec` を継承したクラスを追加し
`@register_model` を付けると、YAMLの `model.name` で指定できるようになる。
