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


_FILENAME_UNSAFE_CHARS = str.maketrans({c: "_" for c in '\\/:*?"<>|'})


def sanitize_filename_component(text: str) -> str:
    """ファイル名に使えない文字（Windowsで禁止されている記号）をアンダースコアに置き換える。

    カラム名やグループ列の値をそのままファイル名の一部にする場合、`/` や `:` などの
    記号が含まれていると保存に失敗するため、保存直前にこの関数でサニタイズする。
    図のタイトル・キャプションに使う文字列は元の値のまま（サニタイズ前）を使う。

    Args:
        text: サニタイズ対象の文字列。

    Returns:
        `\\ / : * ? " < > |` をアンダースコアに置き換えた文字列。
    """
    return text.translate(_FILENAME_UNSAFE_CHARS)
