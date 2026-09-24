"""QuickStart: `src/` 配下のモジュールで「複数系列の時系列予測」の一連の流れを実行するサンプル。

デリーの日次気候データ（`data/raw/DailyDelhiClimateTrain.csv` / `DailyDelhiClimateTest.csv`）の
3つの変数 `meantemp`（日平均気温）・`humidity`（湿度）・`meanpressure`（平均気圧）を、
**1つのモデル（グローバルモデル）で3系列まとめて** 再帰的に多段予測する。
モデルは LightGBM と XGBoost の2種類を作り、最後にアンサンブルする。

グローバルモデルとは:
    変数ごとに別々のモデルを作るのではなく、3変数の履歴を縦に積んだ1つの学習データで
    1つのモデルを学習する方式。「どの変数を予測しているか」は系列ID（`variable` 列）を
    特徴量として渡すことでモデルに伝える。系列間で共通する季節パターンなどを共有して学習でき、
    1系列あたりのデータが少ない場合にも有利。

処理の流れ:
    1. データ読み込みと整形（`util.csv_io.read_csv_auto`）
       - `date` と3つの目的変数以外（`wind_speed`）はドロップ
       - 横持ち（1行 = 1日, 列 = 変数）を縦持ち（1行 = 1日 × 1変数）に変換
    2. 実験設定の作成（`modeling.config.ExperimentConfig`）
       - 外生変数の特徴量（sklearn Pipeline内で学習foldごとに適用）:
         系列IDの数値化 → 日付から月・日を抽出 → 日付列を削除
       - 目的変数の特徴量（系列ごとに計算）: ラグ（1, 2, 3, 7, 14日）と前日までの移動平均（7, 30日）
       - CV: 日付のカットオフで3分割し、各検証期間の先頭114日（テスト期間と同じ長さ）を
         実測値を使わずに再帰予測して評価（再帰バックテスト）。評価指標はMAE
    3. Optunaでハイパーパラメータを探索し、最良パラメータで学習・SHAP解釈・テスト期間の予測
       （`modeling.experiment.run_experiment`）
    4. 2モデルのOOF予測から重みを最適化してアンサンブル（`modeling.ensemble.run_ensemble`）
    5. テスト期間の実測値と比べたMAE（変数別・全体）の表と、予測の比較図を保存

データについての前提:
    - 学習データの最終行（2017-01-01）はテストデータの初日と重なるため、
      テスト開始日以降の学習行は除外する（再帰予測は学習期間の後から始める必要があるため）。
    - テストデータには実測値が含まれるが、予測には使わない
      （予測時は日付と系列IDのみを渡し、予測後の答え合わせだけに使う）。
    - 3変数は尺度が大きく異なる（気温 ≈ 25, 湿度 ≈ 60, 気圧 ≈ 1010）。全体のMAEは
      ばらつきの大きい変数の影響を強く受けるため、変数別のMAEもあわせて確認すること。
    - `meanpressure` には物理的にありえない異常値が含まれる（学習データに 7679, -3, 310 hPa
      など9件、テストデータに 59 hPa が1件。地上気圧はおおむね 950〜1050 hPa）。
      生データは変更せず、このスクリプトの中で次のように扱う:
        * 学習データ: 範囲外の値を欠損にし、前後の日の値から線形補間する
          （そのままだと、ラグ特徴量に異常値が入って再帰予測が大きく乱れるため）
        * テストデータ: 範囲外の実測値は評価（MAEの計算）から除外する
          （予測には実測値を使わないので、予測そのものには影響しない）

Usage:
    uv run python scripts/quickstart_delhi_climate.py
    uv run python scripts/quickstart_delhi_climate.py --n-trials 50
    uv run python scripts/quickstart_delhi_climate.py --n-trials 5 --no-tracking

出力（既定のルートは `outputs/`）:
    - `experiments/delhi_{lightgbm,xgboost}/{実行日時}/`: CVスコア・OOF/テスト予測・
      ステップ別スコア（horizon_scores）・SHAP・チューニング履歴
    - `ensembles/delhi_blend/{実行日時}/`: アンサンブルのスコア・重み・予測
    - `optuna/`: Optuna study（同じコマンドを再実行すると続きから探索する）
    - `tables/delhi_quickstart__test_mae.csv`: テスト期間のMAE（変数別・全体）
    - `figures/delhi_quickstart__test_forecast.png`: テスト期間の実測と予測の比較（変数ごと）
    - MLflow（`mlruns/`）: 実験名 `delhi_quickstart`
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# scripts/ から直接実行しても src/ 配下のパッケージをimportできるようにする
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

# 図はファイル保存のみ行うため非対話型バックエンドに固定する
# （matplotlib.pyplot を最初にimportする前に設定する必要がある）
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from modeling.config import EnsembleConfig, ExperimentConfig
from modeling.ensemble import EnsembleResult, run_ensemble
from modeling.experiment import ExperimentResult, prepare_dataset, run_experiment
from modeling.metrics import get_metric
from modeling.tracking import MLflowTracker, NullTracker, Tracker
from util.csv_io import read_csv_auto
from util.paths import data_dir, ensure_parent_dir, outputs_dir
from util.plotting import add_caption, ensure_japanese_font

# --- 定数 --------------------------------------------------------------------------

# 入力ファイル（data/raw は読み取り専用として扱い、書き込まない）
TRAIN_PATH = data_dir() / "raw" / "DailyDelhiClimateTrain.csv"
TEST_PATH = data_dir() / "raw" / "DailyDelhiClimateTest.csv"

# 予測する3つの変数（それぞれが1つの系列になる）。wind_speed は使わない
TARGET_VARIABLES = ("meantemp", "humidity", "meanpressure")

# 縦持ちに変換した後の列名
TIME_COL = "date"  # 時刻列
SERIES_COL = "variable"  # 系列ID（どの変数の値か）
TARGET = "value"  # 目的変数（各変数の値）

# meanpressure として物理的にありうる範囲（hPa）。範囲外は測定・記録の誤りとみなす
PRESSURE_VALID_RANGE = (950.0, 1050.0)

# 図の軸ラベル用の日本語名と単位
VARIABLE_LABELS = {
    "meantemp": "日平均気温 (℃)",
    "humidity": "湿度 (%)",
    "meanpressure": "平均気圧 (hPa)",
}

MODELS = ("lightgbm", "xgboost")  # 比較するモデル（modeling.models の登録名）
MLFLOW_EXPERIMENT = "delhi_quickstart"  # MLflowで全runをまとめる実験名

# テスト期間の日数。再帰バックテストで評価するステップ数もこれに合わせ、
# 「CVで測る精度」と「本番（テスト期間）で必要な予測の長さ」を揃える
HORIZON = 114


# --- データの読み込みと整形 ---------------------------------------------------------


def to_long(frame: pl.DataFrame) -> pl.DataFrame:
    """横持ち（1行=1日, 列=変数）を縦持ち（1行=1日×1変数）に変換する。

    グローバルモデルでは、3変数を「系列IDの異なる3本の時系列」として縦に積み、
    1つの学習データにする。

    Args:
        frame: `date` と `TARGET_VARIABLES` の列を持つDataFrame。

    Returns:
        列 `date`, `variable`, `value` のDataFrame（日付・変数の順に並べ替え済み）。
    """
    return (
        frame.unpivot(
            index=TIME_COL,
            on=list(TARGET_VARIABLES),
            variable_name=SERIES_COL,
            value_name=TARGET,
        )
        # 同じ日付の中では変数名順に並べる（行順を決めておくと結果が再現しやすい）
        .sort(TIME_COL, SERIES_COL)
    )


def pressure_is_valid() -> pl.Expr:
    """`meanpressure` 列が物理的にありうる範囲内かを表す式（欠損はFalse）。"""
    low, high = PRESSURE_VALID_RANGE
    return pl.col("meanpressure").is_between(low, high).fill_null(False)


def interpolate_invalid_pressure(frame: pl.DataFrame) -> pl.DataFrame:
    """学習データの `meanpressure` の異常値を欠損にし、前後の日の値から線形補間する。

    Args:
        frame: 日付の昇順に並んだ横持ちのDataFrame。

    Returns:
        異常値を補間した DataFrame。
    """
    return frame.with_columns(
        # 範囲外の値を欠損（null）に置き換える
        pl.when(pressure_is_valid())
        .then(pl.col("meanpressure"))
        .otherwise(None)
        # 前後の有効な値を直線で結んで埋める（日付順に並んでいることが前提）
        .interpolate()
        # 先頭・末尾の欠損は直線補間できないので、隣の有効な値で埋める
        .forward_fill()
        .backward_fill()
        .alias("meanpressure")
    )


def is_valid_observation(truth: pl.DataFrame) -> np.ndarray:
    """縦持ちの実測値のうち、評価に使える行かどうか（`meanpressure` の異常値はFalse）。

    Args:
        truth: 縦持ちの実測値（列 `variable`, `value`）。

    Returns:
        評価に使える行ならTrueの真偽値配列。
    """
    low, high = PRESSURE_VALID_RANGE
    is_pressure = pl.col(SERIES_COL) == "meanpressure"
    in_range = pl.col(TARGET).is_between(low, high)
    # 気圧以外の変数は常に有効、気圧は範囲内のときだけ有効
    return truth.select((~is_pressure | in_range).fill_null(False)).to_series().to_numpy()


def load_data(
    train_path: Path = TRAIN_PATH, test_path: Path = TEST_PATH
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """学習データ・予測対象・テスト期間の実測値を、縦持ちの形で読み込む。

    Args:
        train_path: 学習データのCSV。
        test_path: テストデータのCSV。

    Returns:
        (学習データ[date, variable, value],
         予測対象[date, variable]（実測値を含まない）,
         テスト期間の実測値[date, variable, value])。
    """
    # date と3つの目的変数だけを残す（wind_speed はここでドロップされる）
    columns = [TIME_COL, *TARGET_VARIABLES]
    train_wide = read_csv_auto(train_path).select(columns)
    test_wide = read_csv_auto(test_path).select(columns)

    # 学習データの最終日はテストデータの初日と重なっているため、
    # テスト開始日以降の学習行を除外する（再帰予測は学習期間の直後から始めるため）
    test_start = test_wide[TIME_COL].min()
    train_wide = train_wide.filter(pl.col(TIME_COL) < test_start).sort(TIME_COL)

    # 学習データの気圧の異常値は補間する（テストデータの異常値は評価時に除外する）
    train_wide = interpolate_invalid_pressure(train_wide)

    train = to_long(train_wide)
    truth = to_long(test_wide)
    # 予測対象には実測値の列を渡さない（テストの答えがモデルに漏れないようにする）
    test = truth.select(TIME_COL, SERIES_COL)
    return train, test, truth


# --- 設定の作成 ---------------------------------------------------------------------


def make_experiment_config(model: str, n_trials: int) -> ExperimentConfig:
    """1モデル分の実験設定を作る。

    YAMLファイル（`configs/experiments/*.yaml`）に書く場合と同じ内容を、Pythonのdictで
    組み立てて `ExperimentConfig` で検証している。

    Args:
        model: モデル名（`lightgbm` / `xgboost`）。
        n_trials: Optunaの試行回数。

    Returns:
        検証済みの実験設定。
    """
    return ExperimentConfig.model_validate(
        {
            "name": f"delhi_{model}",
            # time_series タスクでは、時間順を無視したCV（kfold等）を指定すると設定エラーになる
            "task": "time_series",
            "data": {
                # 記録用（データはこのスクリプトで読み込んで prepare_dataset に直接渡す）
                "train_path": str(TRAIN_PATH),
                "target": TARGET,
                # 時刻列を指定すると、学習データがこの列で昇順に並べ替えられる
                "time_col": TIME_COL,
            },
            # 外生変数の特徴量。ここに書いた変換は sklearn Pipeline に入り、
            # CVの各foldで「学習期間のデータだけ」を使って fit される（リーク防止）
            "features": [
                # 系列ID（文字列）を整数に変換し、モデルが変数を区別できるようにする
                {
                    "class": "feature_engineering.categorical.PolarsOrdinalEncoder",
                    "params": {"variables": [SERIES_COL]},
                },
                # 日付から月・日を取り出す（date_month, date_day 列が追加される）。
                # 未来の日付でも計算できるので、再帰予測の外生変数として使える
                {
                    "class": "feature_engineering.datetime_features.DatetimeFeaturesExtractor",
                    "params": {"variables": [TIME_COL], "components": ["month", "day"]},
                },
                # 日付そのもの（日時型）はモデルに渡せないため削除する
                {"class": "modeling.pipeline.DropColumns", "params": {"columns": [TIME_COL]}},
            ],
            # 検証期間の開始日。複数系列が混在するデータでは、行番号ではなく
            # 日付で区切る time_cutoff を使う（同じ日の3系列が同じ側に入る）
            "cv": {
                "method": "time_cutoff",
                "cutoffs": ["2015-09-01", "2016-01-01", "2016-05-01"],
            },
            "model": {
                "name": model,
                # 木の本数は多めにし、early stopping で打ち切る
                "params": {"n_estimators": 1000},
                "early_stopping_rounds": 50,
            },
            "metrics": ["mae"],
            # 目的変数の特徴量（再帰予測の設定）。学習時は実測値から、予測時は
            # 予測値を履歴に加えながら、系列（variable）ごとに作り直される
            "forecast": {
                "series_col": SERIES_COL,
                "lags": [1, 2, 3, 7, 14],  # 1〜3日前、1週間前、2週間前の値
                "rolling_windows": [7, 30],  # 前日までの7日・30日の平均
                "horizon": HORIZON,  # 各検証期間の先頭114日を評価する
            },
            # Optunaの探索（探索空間はモデルごとの既定値）。目的関数は再帰バックテストのMAE
            "tuning": {"enabled": True, "n_trials": n_trials},
            # SHAPによる解釈（各foldモデルをその検証期間のデータで説明する）
            "explain": {"enabled": True},
            "tracking": {"experiment_name": MLFLOW_EXPERIMENT},
        }
    )


def make_ensemble_config(member_dirs: dict[str, Path]) -> EnsembleConfig:
    """各実験の出力ディレクトリを構成要素とするアンサンブル設定を作る。

    Args:
        member_dirs: モデル名 → 実験の出力ディレクトリ（OOF・テスト予測を含む）。

    Returns:
        検証済みのアンサンブル設定。
    """
    return EnsembleConfig.model_validate(
        {
            "name": "delhi_blend",
            "task": "time_series",
            # 保存済みの予測ファイルを使うので、モデルの再学習は不要
            "members": [{"name": name, "path": str(path)} for name, path in member_dirs.items()],
            # OOF（再帰バックテストの予測）でMAEが最小になる重み（非負・合計1）を探索する
            "method": "weighted",
            "metrics": ["mae"],
            "tracking": {"experiment_name": MLFLOW_EXPERIMENT},
        }
    )


# --- テスト期間の評価と可視化 --------------------------------------------------------


def evaluate_on_test(truth: pl.DataFrame, predictions: dict[str, np.ndarray]) -> pl.DataFrame:
    """テスト期間の実測値と各モデルの予測から、変数別・全体のMAEを計算する。

    Args:
        truth: テスト期間の実測値（`load_data` の3つ目の戻り値と同じ行順）。
        predictions: モデル名 → 予測値（`truth` と同じ行順）。

    Returns:
        列 `model`, `variable`（`all` は3変数を通した値）, `n`（評価に使った行数）,
        `test_mae` の表。気圧の異常値の行は評価から除外する。
    """
    mae = get_metric("mae")
    y = truth[TARGET].to_numpy()
    series = truth[SERIES_COL].to_numpy()
    # 評価に使える行（気圧の異常値を除く）
    valid = is_valid_observation(truth)
    rows = []
    for model, pred in predictions.items():
        # 変数ごとのMAE（尺度が違うので、比較は同じ変数の中で行う）
        for variable in TARGET_VARIABLES:
            mask = (series == variable) & valid
            rows.append(
                {
                    "model": model,
                    "variable": variable,
                    "n": int(mask.sum()),
                    # 評価できる行が無い変数は欠損（None）にする
                    "test_mae": mae(y[mask], pred[mask]) if mask.any() else None,
                }
            )
        # 3変数を通したMAE（アンサンブルの重み最適化と同じ基準）
        rows.append(
            {
                "model": model,
                "variable": "all",
                "n": int(valid.sum()),
                "test_mae": mae(y[valid], pred[valid]),
            }
        )
    return pl.DataFrame(rows)


def plot_test_forecast(
    truth: pl.DataFrame, predictions: dict[str, np.ndarray], scores: pl.DataFrame
) -> plt.Figure:
    """変数ごとに、テスト期間の実測値と各モデルの予測を重ねて描く。

    Args:
        truth: テスト期間の実測値。
        predictions: モデル名 → 予測値（`truth` と同じ行順）。
        scores: `evaluate_on_test` の結果（凡例にMAEを表示する）。

    Returns:
        作成した図。
    """
    ensure_japanese_font()
    fig, axes = plt.subplots(
        len(TARGET_VARIABLES), 1, figsize=(12, 10), sharex=True, constrained_layout=True
    )
    valid = is_valid_observation(truth)
    for ax, variable in zip(axes, TARGET_VARIABLES, strict=True):
        # この変数の行だけを取り出して描く（行順は日付順）
        mask = (truth[SERIES_COL] == variable).to_numpy()
        dates = truth.filter(pl.col(SERIES_COL) == variable)[TIME_COL].to_list()
        # 評価から除外した異常値はNaNにして線を途切れさせる（縦軸が異常値に引っ張られないように）
        actual = np.where(valid[mask], truth[TARGET].to_numpy()[mask], np.nan)
        ax.plot(dates, actual, color="black", linewidth=2, label="実測")
        for model, pred in predictions.items():
            model_mae = scores.filter(
                (pl.col("model") == model) & (pl.col("variable") == variable)
            )["test_mae"].item()
            ax.plot(dates, pred[mask], alpha=0.85, label=f"{model}（MAE={model_mae:.2f}）")
        ax.set_title(variable)
        ax.set_ylabel(VARIABLE_LABELS[variable])
        # 凡例が線に重ならないよう図の外側（右）に置く
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    axes[-1].set_xlabel("日付")
    fig.suptitle("デリーの気候: テスト期間の再帰予測（2017年1月〜4月、3変数を1つのモデルで予測）")
    add_caption(
        fig,
        f"学習期間の翌日から{truth[TIME_COL].n_unique()}日先までを、予測値をラグに使って再帰予測。"
        f" 気圧の範囲外の実測値（{int((~valid).sum())}件）は描画・評価から除外。",
    )
    return fig


# --- メイン処理 ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """2モデルのチューニング・学習・解釈・アンサンブル・テスト評価を実行する。

    Args:
        argv: コマンドライン引数（Noneなら `sys.argv` を使う。テストから呼ぶ場合に指定）。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=20, help="モデルごとのOptuna試行回数")
    parser.add_argument("--no-tracking", action="store_true", help="MLflowに記録しない")
    parser.add_argument(
        "--output-root", type=Path, default=None, help="出力のルート（既定: outputs/）"
    )
    args = parser.parse_args(argv)
    root = args.output_root or outputs_dir()

    # 1. データの読み込みと整形（縦持ち: 1行 = 1日 × 1変数）
    train, test, truth = load_data()
    print(
        f"学習: {train[TIME_COL].n_unique()}日 × {len(TARGET_VARIABLES)}変数 = {train.height}行"
        f"（{train[TIME_COL].min()}〜{train[TIME_COL].max()}）\n"
        f"予測: {test[TIME_COL].n_unique()}日 × {len(TARGET_VARIABLES)}変数 = {test.height}行"
        f"（{test[TIME_COL].min()}〜{test[TIME_COL].max()}）"
    )

    # 2〜3. モデルごとに「Optuna探索 → 最良パラメータで学習 → SHAP → テスト期間の再帰予測」
    results: dict[str, ExperimentResult] = {}
    for model in MODELS:
        config = make_experiment_config(model, args.n_trials)
        # MLflowに記録しない場合は、何もしない NullTracker を使う
        tracker: Tracker = NullTracker() if args.no_tracking else MLflowTracker(MLFLOW_EXPERIMENT)
        print(f"=== {config.name}: Optuna {args.n_trials}試行 → 学習・SHAP・テスト予測 ===")
        # prepare_dataset: 目的変数の特徴量（ラグ・移動平均）の付与とCV分割の計算
        # run_experiment: チューニング・CV学習・予測の保存・SHAP・MLflow記録をまとめて実行
        results[model] = run_experiment(
            config,
            prepare_dataset(config, train, test),
            tracker=tracker,
            output_root=root / "experiments",
            optuna_dir=root / "optuna",
        )
        cv = results[model].cv_result
        # 再帰予測のMAE（本番に近い評価）と、参考として1日先予測のMAEを並べて表示する。
        # 両者の差が、予測値をラグに使うことによる誤差の蓄積分
        onestep = getattr(cv, "onestep_oof_scores", {}).get("mae", float("nan"))
        print(f"  CV MAE（再帰{HORIZON}日）={cv.oof_scores['mae']:.4f} / 参考: 1日先={onestep:.4f}")
        print(f"  出力: {results[model].output_dir}")

    # 4. アンサンブル（保存済みのOOF予測から重みを求め、テスト予測を加重平均する）
    ensemble_tracker: Tracker = (
        NullTracker() if args.no_tracking else MLflowTracker(MLFLOW_EXPERIMENT)
    )
    ensemble: EnsembleResult = run_ensemble(
        make_ensemble_config({m: r.output_dir for m, r in results.items()}),
        tracker=ensemble_tracker,
        output_root=root / "ensembles",
    )
    print("=== アンサンブル（OOFで重みを最適化） ===")
    print(ensemble.scores)
    if ensemble.weights is not None:
        print("  重み: " + ", ".join(f"{k}={v:.3f}" for k, v in ensemble.weights.items()))

    # テスト期間の予測を集める（いずれも test / truth と同じ行順）
    predictions: dict[str, np.ndarray] = {}
    for model, result in results.items():
        assert result.cv_result.test_pred is not None
        predictions[model] = result.cv_result.test_pred
    assert ensemble.test_pred is not None
    predictions["ensemble"] = ensemble.test_pred

    # 5. テスト期間の実測値との比較（表と図）
    scores = evaluate_on_test(truth, predictions)
    table_path = ensure_parent_dir(root / "tables" / "delhi_quickstart__test_mae.csv")
    scores.write_csv(table_path)
    fig = plot_test_forecast(truth, predictions, scores)
    figure_path = ensure_parent_dir(root / "figures" / "delhi_quickstart__test_forecast.png")
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)  # メモリを解放する

    print("=== テスト期間のMAE（実測との比較） ===")
    excluded = int((~is_valid_observation(truth)).sum())
    print(f"  気圧の範囲外 {PRESSURE_VALID_RANGE} の実測値 {excluded}件 は評価から除外")
    # 行: 変数、列: モデル の表にして表示する
    print(scores.pivot(on="model", index="variable", values="test_mae"))
    print(f"  表: {table_path}")
    print(f"  図: {figure_path}")


if __name__ == "__main__":
    main()
