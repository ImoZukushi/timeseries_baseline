"""画像ギャラリー（Streamlitアプリ）。

指定したディレクトリ内の画像ファイル（PNG・JPEG・GIF・BMP・WebP・SVG）を一覧で表示する。
ファイル名での絞り込み・並べ替え・ページ送りができ、選んだ画像を拡大して詳細（画素数・
サイズ・更新日時）を確認できる。

- `catalog`: 画像を含むディレクトリの探索、画像ファイルの一覧・絞り込み・並べ替え・ページ分割
- `thumbnails`: サムネイルの作成と画像の情報（画素数・形式）の取得
- `main`: Streamlit の画面（起動: `uv run streamlit run app/image_gallery/main.py`）

`main` 以外は Streamlit に依存しない。
"""
