"""時系列ビューア（Streamlitアプリ）。

`data/`・`outputs/` 以下の任意の表形式ファイル（CSV・Parquet・Excel）から時系列を選び、
最大5系列を縦に並べて表示する。クリックした時刻に全系列をまたぐ縦線（カーソル）を引き、
その時刻での各系列の値を比較できる。

- `catalog`: ファイルの一覧と、列の種類（時刻・数値・グループ）の判定
- `storage`: ファイルを LazyFrame として開く（CSV・Excel は Parquet にキャッシュ）
- `series`: 系列の指定・表示範囲の切り出し・間引き・指定時刻の値
- `figure`: Plotly の図（縦に並べた系列・カーソル）
- `main`: Streamlit の画面（起動: `uv run streamlit run app/timeseries_viewer/main.py`）

`main` 以外は Streamlit に依存しない。
"""
