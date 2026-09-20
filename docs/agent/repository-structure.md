# リポジトリ構成

各ディレクトリの役割を説明します。

| ディレクトリ | 役割 |
|-------------|------|
| `src/` | 再利用可能なPythonモジュール |
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
| `docs/agent/` | エージェント向けプロジェクト文書 |
| `.claude/skills/` | 作業別スキルファイル（Claude Code） |
