# `eda` — データ品質チェックと時系列EDA

読み込み済みの polars DataFrame を受け取り、確認用の表と図を保存します。
読み込みは `util.csv_io.read_csv_auto` で行う前提です。

| モジュール | 内容 |
|---|---|
| `eda.csv_quality` | データ全体・列ごとの品質（欠損・重複・分布・上位カテゴリ・相関・記録の並び） |
| `eda.time_series_eda` | 時系列の変換系列（対数・差分・季節差分など）・移動平均・ACF/PACF |

CLI からは `scripts/run_csv_quality_checks.py` で両方をまとめて実行できます。
`data/<サブディレクトリ>` 以下のCSVを走査し、結果を `outputs/figures`・`outputs/tables` に保存します。

```bash
uv run python scripts/run_csv_quality_checks.py                      # data/raw の全CSV
uv run python scripts/run_csv_quality_checks.py --data-subdir processed --pattern "*.csv"
uv run python scripts/run_csv_quality_checks.py --seasonal-period 24 --nlags 72
```

## `eda.csv_quality`

### `run_quality_checks(df, dataset_name, source_path, figures_dir, tables_dir)`

品質確認を一通り行い、表と図を保存します。戻り値は `(全体の要約, 列ごとの要約)` の DataFrame の組です。

| 出力ファイル | 内容 |
|---|---|
| `{tables_dir}/{name}__dataframe_overview.csv` | 行数・列数・重複行・欠損セルの割合など（1行） |
| `{tables_dir}/{name}__column_overview.csv` | 列ごとの型・欠損率・ユニーク数・記述統計・歪度・最頻値 |
| `{figures_dir}/{name}__missing_overview.png` | 列ごとの欠損率 |
| `{figures_dir}/{name}__numeric_histograms.png` | 数値列のヒストグラム |
| `{figures_dir}/{name}__categorical_top_values.png` | 文字列列の上位の値（ユニーク率が高い ID 的な列は除く） |
| `{figures_dir}/{name}__correlation_heatmap.png` | 数値列の相関（欠損のある行を除いた complete case） |
| `{figures_dir}/{name}__record_pattern.png` | 行の並び順に数値列を描いたもの（記録の誤り・外れ値の確認） |
| `{figures_dir}/{name}__scatter_matrix__*.png` | 数値列を3列ずつに分けた組み合わせごとの散布図行列 |

```python
from eda.csv_quality import run_quality_checks
from util.csv_io import read_csv_auto
from util.paths import data_dir, outputs_dir

path = data_dir() / "raw" / "DailyDelhiClimateTrain.csv"
df = read_csv_auto(path)
overview, columns = run_quality_checks(
    df,
    dataset_name="DailyDelhiClimateTrain",
    source_path=str(path),
    figures_dir=outputs_dir() / "figures",
    tables_dir=outputs_dir() / "tables",
)
```

個別の関数も公開しています。いずれも `(df, dataset_name, n_rows, output_path)` を受け取り、図を作れたかを `bool` で返します。

| 関数 | 内容 |
|---|---|
| `dataframe_overview(df, dataset_name, source_path)` | 全体の要約（1行の DataFrame） |
| `column_overview(df, dataset_name)` | 列ごとの要約 |
| `plot_missing_overview` / `plot_numeric_histograms` / `plot_categorical_top_values` / `plot_correlation_heatmap` / `plot_record_pattern` | 上の表の各図 |
| `plot_scatter_matrices(df, dataset_name, n_rows, figures_dir, max_cols_per_image=3)` | 散布図行列をまとめて保存し、パスのリストを返す |
| `complete_case_correlation(df, numeric_cols)` | 欠損のない行だけの相関行列と、使った行数 |
| `make_dataset_name(path, raw_dir)` | `data/raw/wind_0/下条川.csv` → `wind_0_下条川` のような識別名 |

## `eda.time_series_eda`

### 変換系列

1つの数値系列から、次の6種類の系列を作って比べます。

| キー | 系列 |
|---|---|
| `raw` | 原系列 |
| `log` | 対数系列（0以下は欠損） |
| `diff` | 1階差分 |
| `log_diff` | 対数差分 |
| `seasonal_diff` | 季節差分（`seasonal_period` 点前との差） |
| `log_seasonal_diff` | 対数の季節差分 |

変換には `feature_engineering.numeric.LogTransformer` / `feature_engineering.series_transform.DifferenceTransformer` を使います。
そのため、モデリングの目的変数の変換（`forecast.target_transform`）と定義が一致します。

### `run_time_series_checks(df, dataset_name, figures_dir, moving_average_window=5, seasonal_period=7, nlags=40, max_plot_points=5000, max_acf_points=20000)`

日時列を持つ DataFrame について、数値列ごとに次の3枚の図を保存します。
ファイル名の `{base}` は `{dataset_name}__{列名}` です。グループがある場合は `{dataset_name}_{グループ値}__{列名}` になります。

| 出力ファイル | 内容 |
|---|---|
| `{base}__series_with_moving_average.png` | 6系列の値と移動平均（2行3列） |
| `{base}__acf_correlogram.png` | 6系列の ACF |
| `{base}__pacf_correlogram.png` | 6系列の PACF |

戻り値は、図を作った `"{グループ名}:{列名}"` のリストです。

日時列とグループの扱い:
- 日時列（Datetime/Date 型）は自動で探します（`find_datetime_column`）。日時列が無い場合は何もしません。
- 日時が重複している場合（複数地点が混在するパネルデータなど）は、日時と組み合わせて行が一意になる文字列列（`find_grouping_column`）でグループに分けます。
- 1列で一意にならない場合は、誤った自己相関を描かないよう、そのデータは対象外にします。

```python
from eda.time_series_eda import run_time_series_checks
from util.csv_io import read_csv_auto
from util.paths import data_dir, outputs_dir

df = read_csv_auto(data_dir() / "raw" / "weather.csv")  # 地点ごとの時間別データ
done = run_time_series_checks(
    df, "weather", outputs_dir() / "figures", seasonal_period=24, nlags=72
)
print(done[:3])  # 例: ['weather_富山:気温(℃)', ...]
```

### 部品として使う関数

| 関数 | 内容 |
|---|---|
| `find_datetime_column(df)` | 最初の日時型の列名（無ければ None） |
| `find_numeric_columns(df, exclude=())` | 数値型の列名 |
| `find_grouping_column(df, datetime_col)` | 日時の重複を解消できる文字列列（不要・見つからなければ None） |
| `build_transformed_series(values, seasonal_period=7)` | 上の6系列の dict |
| `to_log_series(values)` / `moving_average(values, window=5)` | 対数系列・後方移動平均 |
| `compute_acf(values, nlags=40, max_points=20000)` / `compute_pacf(...)` | ACF / PACF（ラグ0から）。欠損を除いて計算し、点数が足りなければ None |
| `break_line_at_gaps(x, y, gap_factor=10.0)` | 時刻の間隔が中央値の `gap_factor` 倍を超える所で `y` を NaN にする（折れ線の途切れ用） |
| `plot_series_with_moving_average` / `plot_acf_correlogram` / `plot_pacf_correlogram` | 上の各図を1枚ずつ作る |

```python
import polars as pl

from eda.time_series_eda import build_transformed_series, compute_acf

values = pl.Series("y", [float(10 + (i % 7) + i * 0.1) for i in range(200)])
series = build_transformed_series(values, seasonal_period=7)
acf = compute_acf(series["seasonal_diff"], nlags=14)  # 季節差分で周期7の自己相関が消えるかを見る
```

長い系列の ACF/PACF は、計算量を抑えるため直近 `max_acf_points` 点で計算します（間引きはしません）。
