# ac2026
EDA &amp; Baseline

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
