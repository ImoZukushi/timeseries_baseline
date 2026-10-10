"""画像ギャラリーの画面（Streamlit）。

起動:
    uv run streamlit run app/image_gallery/main.py

- サイドバーで、画像を含むディレクトリ（リポジトリ内を自動で探索）を選ぶ
- ファイル名（部分一致・ワイルドカード）・拡張子で絞り込み、名前・更新日時・サイズで並べ替える
- サムネイルを格子状に並べ、「拡大」で元の画像と詳細（画素数・サイズ・更新日時）を表示する

環境変数 `IMAGE_GALLERY_ROOTS`（`os.pathsep` 区切り）で探すディレクトリを変えられる
（既定はリポジトリのルート。`.venv`・`.git` などは除く）。
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run` ではこのファイルのディレクトリしか import パスに入らないため、src と app を加える
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "app"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import os  # noqa: E402

import streamlit as st  # noqa: E402

from image_gallery.catalog import (  # noqa: E402
    ImageDirectory,
    ImageFile,
    SortKey,
    filter_images,
    find_image_directories,
    format_size,
    list_images,
    paginate,
    sort_images,
)
from image_gallery.thumbnails import image_info, make_thumbnail  # noqa: E402

ROOTS_ENV = "IMAGE_GALLERY_ROOTS"
# 最初に開くディレクトリ（あれば）
DEFAULT_DIRECTORY = "outputs/figures"
THUMBNAIL_SIZE = 480
_SORT_LABELS: dict[str, SortKey] = {"名前": "name", "更新日時": "modified", "サイズ": "size"}
_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".webp": "image/webp",
    ".svg": "image/svg+xml",
}


def _roots() -> tuple[str, ...]:
    env = os.environ.get(ROOTS_ENV)
    if env:
        return tuple(p for p in env.split(os.pathsep) if p)
    return (str(_REPO_ROOT),)


# --- 読み込み（Streamlit のキャッシュ）--------------------------------------------------------


@st.cache_data(show_spinner="画像を含むディレクトリを探しています…")
def _directories(roots: tuple[str, ...]) -> list[ImageDirectory]:
    return find_image_directories([Path(r) for r in roots])


@st.cache_data(show_spinner=False, ttl=30)
def _images(directory: str, recursive: bool) -> list[ImageFile]:
    # ファイルの追加・削除を反映するため、30秒でキャッシュを捨てる
    return list_images(Path(directory), recursive=recursive)


@st.cache_data(show_spinner=False, max_entries=2000)
def _thumbnail(path: str, mtime: float, max_size: int) -> bytes | None:
    # 更新日時をキーに含め、画像が書き換わったら作り直す
    return make_thumbnail(Path(path), max_size)


# --- 画面の部品 ------------------------------------------------------------------------------


def _select_directory(directories: list[ImageDirectory]) -> ImageDirectory:
    labels = [f"{d.label}（{d.n_images}枚）" for d in directories]
    default = next((i for i, d in enumerate(directories) if d.label.endswith(DEFAULT_DIRECTORY)), 0)
    index = st.sidebar.selectbox(
        "ディレクトリ",
        range(len(directories)),
        index=default,
        format_func=lambda i: labels[i],
        help="直下に画像があるディレクトリを一覧しています。入力すると候補を絞り込めます",
    )
    return directories[index]


@st.dialog("画像の詳細", width="large")
def _show_detail(image: ImageFile) -> None:
    """選んだ画像を元の大きさで表示し、詳細とダウンロードを出す。"""
    st.image(str(image.path), width="stretch")
    info = image_info(image.path)
    size_text = (
        f"{info.width} × {info.height} px" if info.width is not None else (info.error or "－")
    )
    st.markdown(
        f"**{image.name}**\n\n"
        f"- 画素数: {size_text}\n"
        f"- 形式: {info.format or '－'}"
        + (f"（{info.n_frames} フレーム）" if info.n_frames > 1 else "")
        + f"\n- ファイルサイズ: {format_size(image.size)}\n"
        f"- 更新日時: {image.modified:%Y-%m-%d %H:%M:%S}\n"
        f"- パス: `{image.path}`"
    )
    st.download_button(
        "ダウンロード",
        data=image.path.read_bytes(),
        file_name=image.path.name,
        mime=_MIME.get(image.suffix, "application/octet-stream"),
    )


def _render_card(image: ImageFile) -> None:
    """格子の1枚分（サムネイル・名前・サイズ・拡大ボタン）。"""
    thumb = _thumbnail(str(image.path), image.modified.timestamp(), THUMBNAIL_SIZE)
    with st.container(border=True):
        if thumb is not None:
            st.image(thumb, width="stretch")
        elif image.suffix == ".svg":
            st.image(str(image.path), width="stretch")
        else:
            st.warning("画像を読み込めません")
        st.caption(
            f"**{image.name}**  \n{format_size(image.size)} ・ {image.modified:%Y-%m-%d %H:%M}"
        )
        if st.button("拡大", key=f"open_{image.path}", width="stretch"):
            _show_detail(image)


# --- 画面 ------------------------------------------------------------------------------------


def main() -> None:
    """画像ギャラリーの画面を描く。"""
    st.set_page_config(page_title="画像ギャラリー", layout="wide")
    st.title("画像ギャラリー")

    st.sidebar.header("表示する画像")
    if st.sidebar.button("ディレクトリを探し直す", help="新しく作ったディレクトリを反映します"):
        _directories.clear()
        _images.clear()
    directories = _directories(_roots())
    if not directories:
        st.info("画像（PNG・JPEG・GIF・BMP・WebP・SVG）を含むディレクトリが見つかりません")
        return
    directory = _select_directory(directories)
    recursive = st.sidebar.checkbox("サブディレクトリも含める", value=False)
    images = _images(str(directory.path), recursive)

    st.sidebar.header("絞り込み・並べ替え")
    query = st.sidebar.text_input(
        "ファイル名",
        placeholder="例: qq または *__acf_*.png",
        help="部分一致。* ? でワイルドカード",
    )
    present = sorted({image.suffix for image in images})
    suffixes = st.sidebar.multiselect("拡張子", present, default=present)
    sort_label = st.sidebar.selectbox("並べ替え", list(_SORT_LABELS))
    descending = st.sidebar.checkbox("降順", value=sort_label != "名前")

    st.sidebar.header("表示")
    n_cols = st.sidebar.slider("1行の枚数", 1, 6, 3)
    page_size = st.sidebar.select_slider("1ページの枚数", [6, 12, 24, 48, 96], value=12)

    shown = sort_images(
        filter_images(images, query, suffixes), _SORT_LABELS[sort_label], descending
    )
    st.caption(f"`{directory.path}` ・ {len(shown)} / {len(images)} 枚")
    if not shown:
        st.info("条件に合う画像がありません")
        return

    # 条件が変わったら1ページ目に戻すため、キーに条件を含める
    page_key = f"page_{directory.path}_{recursive}_{query}_{suffixes}_{sort_label}_{page_size}"
    n_pages = paginate(shown, page_size, 1)[1]
    page = 1
    if n_pages > 1:
        page = int(st.number_input(f"ページ（全 {n_pages}）", 1, n_pages, 1, step=1, key=page_key))
    page_items, _ = paginate(shown, page_size, page)

    for start in range(0, len(page_items), n_cols):
        columns = st.columns(n_cols)
        for column, image in zip(columns, page_items[start : start + n_cols], strict=False):
            with column:
                _render_card(image)


main()
