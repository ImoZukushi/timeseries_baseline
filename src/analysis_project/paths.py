"""パス関連のユーティリティ。"""

from __future__ import annotations

from pathlib import Path


def get_repo_root() -> Path:
    """リポジトリルートの `Path` を返す。

    Returns:
        このファイルから2階層上のリポジトリルートパス。
    """
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """`data/` ディレクトリの `Path` を返す。

    Returns:
        リポジトリルート直下の `data` ディレクトリのパス。
    """
    return get_repo_root() / "data"


def outputs_dir() -> Path:
    """`outputs/` ディレクトリの `Path` を返す。

    Returns:
        リポジトリルート直下の `outputs` ディレクトリのパス。
    """
    return get_repo_root() / "outputs"


def ensure_parent_dir(path: Path) -> Path:
    """親ディレクトリを作成してパスを返す。

    Args:
        path: 出力先のパス。

    Returns:
        受け取った `path` をそのまま返す。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
