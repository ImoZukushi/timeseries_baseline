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
同じ内容をセルごとに解説付きで実行できるノートブック版は `notebook/001_quickstart_delhi_climate.ipynb`。
同じデータのEDA（`src/eda` の使い方の例）は `scripts/eda_delhi_climate.py` / `notebook/002_eda_delhi_climate.ipynb`。

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
  clip: {min: 0}            # 予測値の範囲制限（任意）
  target_transform: seasonal_diff  # 目的変数の変換（任意。log / diff / log_diff / seasonal_diff / log_seasonal_diff）
  seasonal_period: 365             # 季節差分の周期（seasonal 系の変換で必須）
```

- `target_transform` を指定すると、モデルは変換後の系列（例: 前年同日との差）を学習・再帰予測し、
  予測値を元の尺度に戻してから評価・出力する。GBDTは学習範囲の外へ外挿できないため、
  トレンドや強い季節性のある系列で有効。変換は `feature_engineering.series_transform` と共通。

- OOF・チューニング・アンサンブルは、検証期間の実測値を使わない再帰予測のスコアで評価する。
  参考として、真のラグを使う1期先予測のスコア（`onestep_oof_*`）もMLflowに記録する。
- ステップ別の誤差は `horizon_scores.csv` / `horizon_error.png` に出力される。
- 前提: 各系列は一定間隔（1行=1ステップ）であり、`test_path` の外生変数は予測時点で既知であること。
  複数系列が混在するデータでは `cv.method: time_cutoff` を推奨。

### 誤差評価の可視化

実験を実行すると、OOF予測から誤差評価の図・表が `{出力}/evaluation/` に自動で保存される
（部品は `src/evaluation/`。sklearn の Display API にならい、Notebook から単独でも使える）。

| タスク | 図 |
|---|---|
| 回帰・時系列 | 残差分布・残差プロット・正規Q-Q・Leverage/Cookの距離・残差のACF/PACF（時刻列がある場合）。複数系列（`forecast.series_col`）では残差分布・Q-Q・ACF/PACFを系列ごとに描く（最大 `evaluation.max_series` 系列） |
| 分類 | 混同行列・ROC曲線・PR曲線（多クラスはOne-vs-Rest） |
| 共通 | 学習の推移（LightGBM・XGBoost・NN）、学習曲線・検証曲線（設定時のみ） |

```yaml
evaluation:
  enabled: true
  learning_curve: {enabled: true, train_sizes: [0.2, 0.5, 1.0]}   # 学習をやり直すため既定は無効
  validation_curve: {param: learning_rate, values: [0.01, 0.05, 0.2]}
```

新しいモデルは `src/modeling/models/` に `ModelSpec` を継承したクラスを追加し
`@register_model` を付けると、YAMLの `model.name` で指定できるようになる。
