# ac2026
EDA &amp; Baseline

## ドキュメント

- [src/ モジュールリファレンス](docs/modules/README.md): `util` / `eda` / `feature_engineering` / `modeling` / `evaluation` の仕様と使い方
- [リポジトリ構成](docs/agent/repository-structure.md)

## 時系列ビューア

`data/`・`outputs/` 以下の任意の時系列データ（CSV・Parquet・Excel）をブラウザで選び、
最大5系列を縦に並べて見比べるアプリ（Streamlit）。

```bash
uv run streamlit run app/timeseries_viewer/main.py
```

- サイドバーで系列ごとにファイル・時刻列・値の列を選ぶ（地点などが混在するファイルはグループ列で1つに絞る）
- 横軸（表示期間・描画点数）と縦軸（範囲・対数軸）はスライダなどで変更できる
- グラフの点をクリックすると、全系列をまたぐ縦線（カーソル）が引かれ、その時刻の各系列の値が表に出る
- 大きなCSVは初回に `data/interim/timeseries_viewer_cache/` へ Parquet のキャッシュを作る（元ファイルは変更しない）

## 画像ギャラリー

リポジトリ内の画像（PNG・JPEG・GIF・BMP・WebP・SVG）をディレクトリごとに一覧表示するアプリ（Streamlit）。

```bash
uv run streamlit run app/image_gallery/main.py
```

- サイドバーで、画像を含むディレクトリ（リポジトリ内を自動で探索。`.venv`・`.git` などは除く）を選ぶ（既定は `outputs/figures`）
- ファイル名（部分一致、`*` `?` のワイルドカード）・拡張子で絞り込み、名前・更新日時・サイズで並べ替える
- サムネイルの格子（1行の枚数・1ページの枚数を変更可）から「拡大」で元の画像・画素数・更新日時を表示し、ダウンロードできる
