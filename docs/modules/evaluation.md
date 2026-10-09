# `evaluation` — 誤差評価の可視化

予測値・実測値（必要なら特徴量行列・SHAP値）を受け取って、図と表を作る部品です。`modeling` に依存しないので、ノートブックで単独で使えます。

`modeling` の実験では、これらの部品が自動で呼ばれます（[実験から自動で作られるもの](#実験から自動で作られるもの)）。

| モジュール | 内容 |
|---|---|
| `residuals` | 残差の要約・残差分布・残差プロット・正規Q-Q・残差のACF/PACF |
| `time_series_diagnostics` | 時系列の残差診断（残差の推移・Ljung-Box・Jarque-Bera・ADF/KPSS・診断パネル） |
| `influence` | Leverage と Cook の距離（影響の大きいサンプル） |
| `classification` | 混同行列・ROC曲線・PR曲線（二値・多クラス） |
| `curves` | 学習曲線・検証曲線・学習の推移・予測ステップ別の誤差 |
| `shap_correlation` | 特徴量の値と SHAP 値の相関・特徴量同士の SHAP 値の相関 |

## Display の共通の使い方

scikit-learn の Display API に合わせています。

- **計算と描画:** `XxxDisplay.from_predictions(...)`（入力によって `from_residuals` / `from_shap` など）で計算して描きます。
- **描き直し:** `plot(ax=...)` で別の Axes に描き直せます。
- **描画後の属性:** `figure_` / `ax_` と、計算結果の属性を持ちます。
- **既存の Axes に描く:** `ax` を渡すとその Axes に描きます。2枚組の図には、Axes を2個並べて渡します。
- **残差の向き:** 残差は `実測値 − 予測値` です（sklearn の `PredictionErrorDisplay` と同じ）。欠損の行は除いて計算します。

```python
import matplotlib.pyplot as plt
import numpy as np

from evaluation.residuals import QQPlotDisplay, ResidualDistributionDisplay
from util.paths import ensure_parent_dir, outputs_dir

rng = np.random.default_rng(0)
y_true = rng.normal(size=500)
y_pred = y_true + rng.standard_t(4, size=500) * 0.3

disp = ResidualDistributionDisplay.from_predictions(y_true, y_pred)
print(disp.summary)  # n, mean, std, skewness, excess_kurtosis, mae, rmse
path = ensure_parent_dir(outputs_dir() / "figures" / "doc_residual_distribution.png")
disp.figure_.savefig(path, dpi=150, bbox_inches="tight")
plt.close(disp.figure_)

# 既存の Axes に並べて描く
fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
ResidualDistributionDisplay.from_predictions(y_true, y_pred, ax=axes[0])
QQPlotDisplay.from_residuals(y_true - y_pred, ax=axes[1])
plt.close(fig)
```

## `residuals`

| 名前 | 内容 |
|---|---|
| `residual_summary(y_true, y_pred)` | 件数・平均・標準偏差・歪度・尖度（正規分布で0）・MAE・RMSE の dict |
| `ResidualDistributionDisplay` | 残差のヒストグラムと、同じ平均・標準偏差の正規分布。`from_predictions` / `from_residuals` |
| `ResidualPlotDisplay` | 2枚組: 残差 vs 予測値 ／ 実測値 vs 予測値（sklearn の `PredictionErrorDisplay`）。`max_points` で間引く |
| `QQPlotDisplay` | 標準化した残差の正規 Q-Q プロット。`r`（当てはめ直線の相関）を持つ。`from_predictions` / `from_residuals` |
| `ResidualCorrelogramDisplay` | 残差の ACF・PACF（系列ごとに1段）。残差は**時刻順**に渡す。`from_residuals({系列名: 残差})` |
| `plot_residuals_by_group(kind, groups, ncols=2)` | 系列ごとの残差分布（`kind="distribution"`）または Q-Q（`"qq"`）をパネルに並べる。`groups` は `{系列名: (y_true, y_pred)}` |

尺度の違う複数の系列の残差を1つの分布にまとめると、系列間の差が裾の重さのように見えます。複数系列では `plot_residuals_by_group` で系列ごとに見てください。

## `time_series_diagnostics`

時刻順の残差が白色雑音（偏り・自己相関がなく、ばらつきが一定）に近いかを、図と検定で調べます。

| 検定 | 帰無仮説 | 関数 |
|---|---|---|
| Ljung-Box | 指定ラグまで自己相関なし | `ljung_box_test(residuals, lags=None, *, seasonal_period=None, model_df=0)` → 表（`lag`, `statistic`, `p_value`） |
| Jarque-Bera | 正規分布 | `jarque_bera_test(residuals)` → `statistic`, `p_value`, `skewness`, `excess_kurtosis` |
| ADF ＋ KPSS | ADF: 単位根あり ／ KPSS: 定常 | `unit_root_test(residuals, *, regression="c", alpha=0.05)` → 統計量・p 値・`conclusion` |

- **Ljung-Box のラグ:** 既定は `min(10, n//5)` です。`seasonal_period` が点数の半分未満なら、その周期も加えます。点数の半分以上のラグは使いません。
- **`model_df`（自由度補正）:** 機械学習モデルでは定義できないため 0 です。ARIMA などでは AR と MA の次数の和を渡します。
- **KPSS の p 値:** 統計表の範囲（0.01〜0.1）で打ち切られます。
- **単位根の判定（`conclusion`）:** ADF と KPSS の結果を組み合わせて決めます。

| ADF | KPSS | 判定 |
|---|---|---|
| 棄却 | 棄却せず | 定常 |
| 棄却せず | 棄却 | 非定常（単位根の疑い） |
| 棄却せず | 棄却せず | 判定が分かれる（データ不足・検出力不足の可能性） |
| 棄却 | 棄却 | 判定が分かれる（構造変化・差分定常の可能性） |

点数が足りない・値が一定などで計算できないときは、例外にせず NaN（判定は「計算できない」）を返します。

| まとめ・図 | 内容 |
|---|---|
| `residual_tests(residuals, *, lags, seasonal_period, regression, alpha)` | 系列ごとに3検定をまとめる → `(要約表, Ljung-Box の全ラグの表)`。要約表は判定列 `autocorrelation`（あり/なし）・`normality`（棄却/棄却せず）・`stationarity` を持つ |
| `ResidualTimeSeriesDisplay.from_residuals(residuals, *, time, segments, rolling_window)` | 残差の推移（系列ごとに1段）。移動平均・±2σ の帯を重ねる。`segments`（fold 番号など）の境目と時刻の大きな空白では線を切る |
| `TimeSeriesResidualDiagnosticsDisplay.from_residuals(residuals, *, time, segments, nlags, lags, seasonal_period, ...)` | 1系列の診断を1枚に: 推移・ヒストグラム・Q-Q・ACF・PACF・検定結果の表。`tests` / `ljung_box` 属性に検定結果 |

```python
import matplotlib.pyplot as plt
import numpy as np

from evaluation.time_series_diagnostics import TimeSeriesResidualDiagnosticsDisplay, residual_tests

rng = np.random.default_rng(0)
white = rng.normal(size=400)
ar1 = np.zeros(400)
for t in range(1, 400):
    ar1[t] = 0.7 * ar1[t - 1] + rng.normal()

summary, ljung_box = residual_tests({"white": white, "ar1": ar1}, seasonal_period=7)
print(summary.select("series", "ljung_box_p_value", "autocorrelation", "normality", "stationarity"))

disp = TimeSeriesResidualDiagnosticsDisplay.from_residuals(ar1, seasonal_period=7, title="AR(1) の残差")
plt.close(disp.figure_)
```

多段先（再帰）予測の残差は、誤差が蓄積するため自己相関があるのが自然です。モデルの当てはまりは、1期先予測の残差で診断してください。
CV の fold をつないだ残差では、fold の境目も連続した時系列として扱われます。

## `influence`

| 名前 | 内容 |
|---|---|
| `compute_influence(X, residuals)` | サンプルごとの `row`（入力の行番号）・`leverage`・`residual`・`standardized_residual`・`cooks_distance` の表 |
| `InfluenceDisplay.from_predictions(X, y_true, y_pred, *, max_points=5000, n_labels=5)` | 2枚組: Leverage vs 標準化残差（Cook の距離の等高線つき）／ Cook の距離。`top(k)` で Cook の距離の上位 k 行 |

- **Leverage:** ハット行列 `X(XᵀX)⁻¹Xᵀ` の対角成分です。特徴量の組み合わせが外れているサンプルほど大きくなります。
- **Cook の距離:** そのサンプルを除いて学習し直したときの、予測の変わり方の目安です。`4/n` や 0.5 を超えるものが注意の目安です。

本来は線形回帰の診断量です。線形回帰では statsmodels の `OLSInfluence` と一致します。
GBDT などでは、「特徴量が外れていて、かつ誤差が大きいサンプル」を探す目安として使います。

## `classification`

入力の形: `y_true` はクラス番号（0..K-1）、`y_score` は二値なら陽性の確率 (n,)、多クラスなら確率 (n, K) です。

| 名前 | 内容 |
|---|---|
| `ConfusionMatrixDisplay.from_predictions(y_true, y_score, *, class_names=None, threshold=0.5)` | 2枚組: 件数 ／ 実測クラスごとの割合（再現率）。二値は `threshold` 以上を陽性、多クラスは確率最大のクラス |
| `RocCurveDisplay.from_predictions(y_true, y_score, *, class_names=None)` | ROC 曲線（凡例に AUC）。多クラスは One-vs-Rest で各クラスの曲線 |
| `PrecisionRecallDisplay.from_predictions(...)` | PR 曲線（凡例に AP、点線は陽性率） |
| `scores_to_labels(y_score, threshold=0.5)` | 確率をクラス番号にする |

```python
import matplotlib.pyplot as plt
import numpy as np

from evaluation.classification import ConfusionMatrixDisplay, RocCurveDisplay

rng = np.random.default_rng(0)
y_true = rng.integers(0, 3, 300)
proba = rng.dirichlet(np.ones(3), 300)
proba[np.arange(300), y_true] += 0.5           # 正解クラスの確率を高めた疑似的な予測
proba /= proba.sum(axis=1, keepdims=True)

for disp in (
    ConfusionMatrixDisplay.from_predictions(y_true, proba, class_names=["低", "中", "高"]),
    RocCurveDisplay.from_predictions(y_true, proba, class_names=["低", "中", "高"]),
):
    plt.close(disp.figure_)
```

## `curves`

| 名前 | 内容 |
|---|---|
| `compute_learning_curve(estimator, X, y, cv, scoring, *, train_sizes, negate=False, time_ordered=False)` | 学習データの量を変えたときの学習・検証スコア（sklearn の `learning_curve`）→ `CurveResult` |
| `compute_validation_curve(estimator, X, y, cv, scoring, *, param_name, param_range, negate=False)` | 1つのパラメータを動かしたときの学習・検証スコア（sklearn の `validation_curve`）→ `CurveResult` |
| `plot_learning_curve(result, ax=None, *, title="")` / `plot_validation_curve(result, param_name, ...)` | 上の結果を描く（sklearn の Display） |
| `most_recent_first(folds)` | 時系列の学習曲線用。各 fold の学習行を新しい順に並べ替え、少ないデータでは直近の期間を使うようにする |
| `TrainingHistory` / `TrainingHistoryDisplay.from_histories(histories)` | 木の本数・エポックごとの学習・検証の損失（fold ごとに1組の線、最良反復に縦線） |
| `HorizonErrorDisplay.from_scores(horizon_scores, metric, *, reference=None)` | 再帰予測の予測ステップ別の誤差（`reference` に1期先のスコアを横線で） |

- **`cv` に渡すもの:** `(学習の行番号, 検証の行番号)` の組のリストを渡せます。`modeling` の CV 分割をそのまま使えます。
- **`negate=True`:** 誤差系の scorer（`neg_*`）の符号を戻して、誤差として表示します。

## `shap_correlation`

入力は SHAP 値 (n, 特徴量数)・特徴量の値 (n, 特徴量数)・特徴量名です。多クラスは、クラスごとに (n, 特徴量数) を切り出して渡します。

| 名前 | 内容 |
|---|---|
| `shap_feature_correlation(shap_values, data, feature_names)` | 特徴量ごとの平均\|SHAP\|と、特徴量の値と SHAP 値の相関（Pearson・Spearman）・向き（正/負/なし） |
| `ShapCorrelationBarDisplay.from_shap(...)` | 平均\|SHAP\|の棒グラフを向きで色分け（赤＝値が大きいほど予測を上げる、青＝下げる） |
| `ShapDependenceDisplay.from_shap(..., top_k=6)` | 重要度上位の「特徴量の値 vs SHAP 値」の散布図。色は相互作用が最も強そうな別の特徴量（`shap.utils.approximate_interactions`） |
| `shap_value_correlation(shap_values, feature_names)` / `ShapValueCorrelationDisplay.from_shap(..., top_k=15)` | 特徴量同士の SHAP 値の相関行列（赤＝同じ向きに効く・冗長の可能性、青＝打ち消し合う） |
| `ShapScatterMatrixDisplay.from_shap(..., top_k=5)` | SHAP 値同士の散布図行列（下三角: 散布図と相関係数、対角: 分布） |

`modeling.explain.ShapResult` から使う場合は、次のように渡します。

```python
from evaluation.shap_correlation import ShapDependenceDisplay

disp = ShapDependenceDisplay.from_shap(
    shap_result.values,                                       # (n, 特徴量数)
    shap_result.data.to_numpy(dtype=float, na_value=float("nan")),
    shap_result.feature_names,
)
```

## 実験から自動で作られるもの

`modeling.experiment.run_experiment` は `evaluation.enabled: true`（既定）のとき、OOF 予測から次のファイルを `{実験の出力}/evaluation/` に保存します。
OOF 予測とは、各 fold のモデルが、学習に使っていない検証データに対して出した予測です。

| 条件 | ファイル |
|---|---|
| 回帰・時系列 | `residual_summary.csv`（全体＋系列ごと）, `residual_distribution.png`, `qq_plot.png`, `residual_plot.png`, `leverage_cooks_distance.png`, `top_cooks_distance.csv` |
| 時刻列あり（`data.time_col`） | `residual_acf_pacf.png`, `residual_tests.csv`, `ljung_box.csv`, `residual_timeseries.png`, `residual_diagnostics__{種類}[__{系列}].png` |
| 再帰予測（`forecast`） | 上記の時系列の残差診断を、再帰予測（`recursive`）と1期先予測（`onestep`。`residual_timeseries_onestep.png`）の両方の残差で作る |
| 二値・多クラス | `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png` |
| 学習の推移の記録がある（LightGBM・XGBoost・NN） | `training_history.png` |
| `learning_curve.enabled` / `validation_curve.param` | `learning_curve.png` / `.csv`, `validation_curve.png` / `.csv` |

複数系列（`forecast.series_col`）では、残差分布・Q-Q・ACF/PACF・診断パネルを系列ごとに描きます（最大 `evaluation.max_series` 系列）。
`residual_tests.csv` には全系列の結果が入ります。
