# `util` — 汎用ヘルパー

パス、CSVの読み込み、matplotlibの日本語設定といった、どのパッケージからも使う小さな部品です。

| モジュール | 内容 |
|---|---|
| `util.paths` | リポジトリ内の主要ディレクトリのパス |
| `util.csv_io` | エンコーディングを自動判定するCSV読み込み（日時列の自動変換・欠測記号の扱い） |
| `util.plotting` | matplotlib の日本語フォント設定・図のキャプション |

## `util.paths`

| 関数 | 戻り値 |
|---|---|
| `get_repo_root()` | リポジトリのルート |
| `data_dir()` | `data/` |
| `outputs_dir()` | `outputs/` |
| `ensure_parent_dir(path)` | 親ディレクトリを作成し、`path` をそのまま返す |
| `sanitize_filename_component(text)` | ファイル名に使えない記号（`\ / : * ? " < > \|`）を `_` に置き換える |

```python
from util.paths import data_dir, ensure_parent_dir, outputs_dir, sanitize_filename_component

train_path = data_dir() / "raw" / "DailyDelhiClimateTrain.csv"
# 列名などをファイル名に使うときは、禁止文字を置き換えてから使う
out_path = ensure_parent_dir(
    outputs_dir() / "tables" / f"{sanitize_filename_component('風速(m/s)')}.csv"
)
```

## `util.csv_io`

### `read_csv_auto(path) -> pl.DataFrame`

CSVを読み込み、次の処理をまとめて行います。

- **エンコーディング:** UTF-8 で読めなければ CP932（Shift-JIS）として読みます。
- **日時列:** 文字列の列のうち、決まった書式に95%以上一致する列を Datetime 型に変換します。
  対応する書式は `%Y-%m-%dT%H:%M:%S%.f`（ISO 形式。polars の `write_csv` が書き出す形）/ `%Y-%m-%d %H:%M:%S` / `%Y-%m-%d` /
  `%Y/%m/%d %H:%M:%S` / `%Y/%m/%d %H:%M` / `%Y/%m/%d` です。
- **欠測記号:** `*` を欠損（null）として読みます。`wind_*.csv` の風速・風向で使われています。
- **大容量ファイル:** 30MBを超え、ヘッダ以外がASCII文字だけのファイルは、全文をデコードせずに読みます。

### `scan_csv_auto(path, datetime_sample_rows=1000) -> pl.LazyFrame`

`read_csv_auto` の LazyFrame 版です。大容量でデータ本体がASCII文字だけのCSVは、`pl.scan_csv` で遅延読み込みします。
`sink_parquet` などと組み合わせれば、全行をメモリに載せずに処理できます。それ以外のファイルは `read_csv_auto(path).lazy()` と同じです。

`read_csv_auto` と違い、日時の書式は先頭 `datetime_sample_rows` 行だけで判定します。
時系列ビューア（`app/timeseries_viewer`）の Parquet キャッシュ作成で使っています。

```python
import polars as pl

from util.csv_io import read_csv_auto, scan_csv_auto
from util.paths import data_dir

weather = read_csv_auto(data_dir() / "raw" / "weather.csv")  # CP932・日時列は自動変換

# 580MBの秒単位データも、全行を読まずに集計できる
wind = scan_csv_auto(data_dir() / "raw" / "wind_0" / "下条川.csv")
daily_max = (
    wind.group_by(pl.col("年月日時").dt.date().alias("date"))
    .agg(pl.col("風速(瞬時)").max())
    .collect()
)
```

その他の公開関数・定数:

| 名前 | 内容 |
|---|---|
| `detect_encoding(path, sample_size=1_000_000)` | 先頭のバイト列から `"utf-8"` か `"cp932"` を推定する |
| `LARGE_FILE_THRESHOLD_BYTES` | 大容量ファイルとみなす閾値（30MB） |
| `NULL_VALUE_MARKERS` | 欠測とみなす値（`["*"]`） |
| `DATETIME_PARSE_SUCCESS_THRESHOLD` | 日時列とみなす変換成功率（0.95） |

## `util.plotting`

| 関数 | 内容 |
|---|---|
| `ensure_japanese_font()` | 日本語フォント（Windows標準の Meiryo）とマイナス記号の表示を設定する。`sns.set_theme()` はフォントを初期化するので、テーマ設定の**後に**呼ぶ |
| `add_caption(fig, text)` | 図の下部に注記（データ期間・件数など）を入れる |

```python
import matplotlib.pyplot as plt
import seaborn as sns

from util.plotting import add_caption, ensure_japanese_font

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.2)
ensure_japanese_font()  # set_theme の後に呼ぶ
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
ax.plot([1, 2, 3], [2, 4, 3])
ax.set_title("日本語の見出し")
add_caption(fig, "n=3, 2024年1月〜3月")
plt.close(fig)
```

`eda`・`evaluation`・`modeling` の描画関数は内部で `ensure_japanese_font()` を呼ぶので、呼び出し側での設定は不要です。
japanize-matplotlib は Python 3.12 で import できない（distutils 依存）ため使っていません。Meiryo の無い環境（macOS・Linux）では、
`plt.rcParams["font.family"]` に手元の日本語フォント（Hiragino Sans・Noto Sans CJK JP など）を指定してください。
