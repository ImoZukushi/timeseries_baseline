# テスト・検証方針

## pytest

- テストは `tests/` ディレクトリに配置する
- `uv run pytest` で実行する
- テストは高速に保つ（外部依存を最小限に）

## ruff

- `uv run ruff check .` でリントする
- `uv run ruff format .` でフォーマットする
- 設定は `pyproject.toml` の `[tool.ruff]` セクション

## mypy

- `uv run mypy src` で型チェックする
- 設定は `pyproject.toml` の `[tool.mypy]` セクション

## Notebook検証

- Notebookがクリーンなカーネルから再実行できることを確認する
- 秘密情報がセル出力に含まれていないことを確認する

