# `feature_engineering` — 特徴量の transformer

polars DataFrame を受け取って polars DataFrame を返す、sklearn 互換の transformer の集まりです。
`fit` / `transform`（`fit_transform` は `TransformerMixin`）の規約に従うので、`sklearn.pipeline.Pipeline` にそのまま入れられます。
`modeling` の実験設定 `features` に書くと、CV の fold ごとに学習データだけで fit されます。

| モジュール | 内容 | 列の扱い |
|---|---|---|
| `categorical` | カテゴリ変数のエンコーディング | 置き換え（ワンホットのみ列を追加） |
| `datetime_features` | 日時の成分・経過時間・周期特徴量 | 列を追加（元の日時列は残る） |
| `numeric` | 数値の変換（対数・逆数・平方根・べき乗・Box-Cox/Yeo-Johnson） | 置き換え |
| `scaling` | スケーリング（最小最大・ロバスト・平均正規化・最大絶対値） | 置き換え |
| `time_series` | ラグ・移動平均・変化率 | 列を追加 |
| `series_transform` | 逆変換できる時系列の変換（対数・差分・季節差分など） | 置き換え |

## 共通の仕様

- **対象列:** コンストラクタの `variables` に、列名1つまたは列名のリストで指定します。
- **学習した値:** `fit` で学習した値は `xxx_`（末尾がアンダースコア）の属性に保存されます。`transform` はその値だけを使います。
- **リーク防止:** `fit` は学習データにだけ呼び、検証・テストデータには `transform` だけを呼びます。
- **列の扱い:**
  - 「置き換え」の transformer は、同じ列名のまま値を変換します。多くは `inverse_transform` で元に戻せます。
  - 「列を追加」の transformer は、元の列を残して `{列名}_{接尾辞}` の列を加えます。
  - モデルに入れる前に、文字列・日時の列はエンコードするか `modeling.pipeline.DropColumns` で削除してください。
- **時系列の前提:** 時系列系の transformer は、入力が時刻の昇順に並んでいる前提です（並べ替えはしません）。
  複数の系列が混ざるデータは `group_by` に系列IDの列を渡すと、系列ごとに計算します。

```python
import polars as pl
from sklearn.pipeline import Pipeline

from feature_engineering.categorical import PolarsOrdinalEncoder
from feature_engineering.datetime_features import CyclicalFeaturesEncoder, DatetimeFeaturesExtractor
from feature_engineering.time_series import LagFeatureGenerator

train = pl.DataFrame(
    {
        "date": pl.datetime_range(
            pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 10), "1d", eager=True
        ),
        "store": ["a", "b"] * 5,
        "sales": [float(v) for v in range(10)],
    }
)
pipe = Pipeline(
    [
        ("store", PolarsOrdinalEncoder("store")),
        ("date", DatetimeFeaturesExtractor("date", components=["month", "weekday"])),
        ("cyclic", CyclicalFeaturesEncoder("date_weekday", period=7)),
        ("lag", LagFeatureGenerator("sales", lags=[1, 2], group_by="store")),
    ]
)
features = pipe.fit_transform(train)
# 追加される列: date_month, date_weekday, date_weekday_sin, date_weekday_cos, sales_lag_1, sales_lag_2
```

## `categorical`

| クラス | 主な引数 | 出力 | 未知のカテゴリ |
|---|---|---|---|
| `PolarsOneHotEncoder` | `min_frequency`, `max_categories` | `{列名}_{カテゴリ}` の0/1列を追加（元の列は残る） | すべて0 |
| `PolarsOrdinalEncoder` | － | カテゴリを整数に置き換え | null |
| `TargetGuidedOrdinalEncoder` | － | 目的変数の平均が小さい順の順位に置き換え（`fit(X, y)`） | null |
| `PolarsTargetEncoder` | `smoothing`, `cv=5`, `target_type`, `random_state` | 目的変数の平均（sklearn `TargetEncoder`）に置き換え | 学習時の全体平均 |
| `RareLabelGrouper` | `min_frequency` または `min_ratio`, `other_label="Other"` | 頻度の低いカテゴリを `other_label` にまとめる | `other_label` |

`PolarsTargetEncoder.fit_transform(X, y)` は、学習データ自身を cross-fitting（他の fold の統計量でエンコード）で変換します。
`fit(X, y).transform(X)` と結果が異なり、学習データの目的変数がそのまま漏れません。
Pipeline の中では `fit_transform` が使われるので、意識せずに安全に使えます。

## `datetime_features`

| クラス | 主な引数 | 追加される列 |
|---|---|---|
| `DatetimeFeaturesExtractor` | `components`（`year` / `month` / `day` / `weekday` / `week` / `hour` / `minute` / `second`） | `{列名}_{成分}` |
| `ElapsedTimeTransformer` | `unit`（`seconds` / `minutes` / `hours` / `days`） | `{列名}_elapsed_{unit}`（学習データの最小時刻からの経過） |
| `CyclicalFeaturesEncoder` | `period`（例: 月なら12、曜日なら7、時なら24） | `{列名}_sin`, `{列名}_cos` |

`CyclicalFeaturesEncoder` には数値の列（`DatetimeFeaturesExtractor` が作った `date_month` など）を渡します。

## `numeric`

| クラス | 変換 | 逆変換 | 範囲外の値 |
|---|---|---|---|
| `LogTransformer(base=e, offset=0)` | `log(x + offset)` | `base ** y - offset` | `x + offset <= 0` は null |
| `ReciprocalTransformer` | `1 / x` | `1 / y` | `x = 0` は null |
| `SqrtTransformer` | `√x` | `y²` | 負の値は null |
| `FixedPowerTransformer(power)` | `x ** power` | `y ** (1/power)` | － |
| `PolarsPowerTransformer(method="yeo-johnson")` | Box-Cox / Yeo-Johnson（λ を学習） | 学習した λ で逆変換 | Box-Cox は正の値のみ |

## `scaling`

すべて学習データの統計量で変換し、`inverse_transform` で元に戻せます。

| クラス | 変換 |
|---|---|
| `PolarsMinMaxScaler` | `(x - min) / (max - min)` |
| `PolarsRobustScaler(quantile_range=(25, 75))` | `(x - 中央値) / 四分位範囲` |
| `MeanNormalizationScaler` | `(x - mean) / (max - min)` |
| `PolarsMaxAbsScaler` | `x / max(|x|)` |

## `time_series`

未来の値を参照しないよう、正のラグと後方の移動平均だけを提供します（中心化した移動平均や負のラグはありません）。

| クラス | 主な引数 | 追加される列 |
|---|---|---|
| `LagFeatureGenerator` | `lags`（1以上）, `group_by` | `{列名}_lag_{k}`（先頭 k 行は null） |
| `MovingAverageTransformer` | `window`, `min_periods`, `group_by` | `{列名}_ma_{window}`（現在の行を含む後方の平均） |
| `RateOfChangeTransformer` | `periods=1`, `group_by` | `{列名}_roc_{periods}`（前期比。分母0は null） |

`MovingAverageTransformer` は**現在の行を含む**平均です。目的変数の移動平均を特徴量にするときは、ラグ列（`y_lag_1`）に対して使います。
`modeling` の `forecast.rolling_windows` は、内部でこの形（`{目的変数}_lag_1_ma_{w}`）にしています。

## `series_transform`

トレンド・季節性を取り除いた系列で学習し、予測値を元の尺度に戻すための変換です。`make_series_transformer(kind, variables, ...)` で名前から作れます。

| `kind` | 変換 | 作られるクラス |
|---|---|---|
| `log` | `log(x + offset)` | `numeric.LogTransformer` |
| `diff` | `x_t - x_{t-1}` | `DifferenceTransformer(periods=1)` |
| `log_diff` | `log x_t - log x_{t-1}` | `DifferenceTransformer(periods=1, log=True)` |
| `seasonal_diff` | `x_t - x_{t-s}` | `DifferenceTransformer(periods=s)` |
| `log_seasonal_diff` | `log x_t - log x_{t-s}` | `DifferenceTransformer(periods=s, log=True)` |

`DifferenceTransformer` の仕様:
- 各系列の先頭 `periods` 行は前の値が無いため null になります。
- `log=True` で `x + offset <= 0` の値があると、`fit` でエラーにします（黙って null にすると逆変換が壊れるため）。
- 逆変換 `inverse_transform(X, history=...)` は、差分を累積して元の値に戻します。起点は `history` で選びます。

| `history` | 起点 | 用途 |
|---|---|---|
| `None`（既定） | fit したデータの先頭 | fit したデータと同じ期間を復元する（Pipeline の `inverse_transform`） |
| `"fit"` | fit したデータの末尾 | fit したデータの**直後**の期間（予測値）を復元する |
| DataFrame | 渡した過去データ（元の尺度）の末尾 | その直後の期間を復元する |

```python
import polars as pl

from feature_engineering.series_transform import make_series_transformer

history = pl.DataFrame({"y": [float(10 + i % 7) for i in range(28)]})
tf = make_series_transformer("seasonal_diff", "y", seasonal_period=7)
z = tf.fit_transform(history)  # 先頭7行は null、以降は前週同曜日との差

# 予測した差分（直後の7期）を元の尺度に戻す
z_pred = pl.DataFrame({"y": [0.0] * 7})
y_pred = tf.inverse_transform(z_pred, history="fit")  # 前週と同じ値になる
```

差分の途中に欠損があると、それ以降の同じ位相の値は復元できず null になります（差分表現の原理的な制約です）。
