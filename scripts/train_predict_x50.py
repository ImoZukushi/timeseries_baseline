"""前処理済みの x50 データで目的変数 Y1〜Y10 のモデルを学習し、テスト期間（2025年）を再帰予測する。

入力は `notebook/003_feature_engineering_x50.ipynb` が出力した次の2ファイル。

- `data/processed/x50/x50_train_features.csv`: 学習データ（目的変数 `Yk`（対数）・`Yk_raw`・
  目的変数ごとのラグ等・日付の特徴量・説明変数）
- `data/processed/x50/x50_test_features.csv`: テストデータ（日付の特徴量・説明変数。目的変数なし）

目的変数ごとに別のモデルを `modeling.experiment.run_experiment` で学習する
（LightGBM / XGBoost から選ぶ）。

- 目的変数 Yk のモデルの特徴量は「Yk 自身のラグ・移動平均・前日比 + 日付 + 説明変数」。
  他の目的変数のラグは使わない（テスト期間では他の目的変数の値も分からないため）。
- 目的変数は `Yk_raw`（元の尺度）とし、`forecast.target_transform="log"` でモデルには対数で
  学習させる。評価指標（MAE）・予測値は元の尺度。
- 目的変数のラグ・移動平均・前日比は、学習時は過去の実測値から、検証・テスト期間は再帰予測の中で
  予測値を履歴に足しながら作り直す（`modeling.forecasting`）。学習データのCSVにある同じ列は、
  作り直した値と一致することを確かめたうえで使わない。
- CVは日付で区切る時系列CV（`time_cutoff`）で、検証期間も再帰予測で評価する
  （検証期間の実測値を使わない）。
- 正解ファイル（`data/synthetic/x50/dummy_x50_test_answer.csv`）は、テスト予測が終わった後の
  評価にだけ使う。

出力:
    - 目的変数ごと（`outputs/experiments/x50_{モデル名}_{目的変数}/{実行日時}/`）:
      `run_experiment` の出力一式（CVスコア・OOF/テスト予測・誤差評価の図・SHAPの図・設定）、
      `test_predictions.csv`、`test_scores.csv`、`figures/training_result.png`、`figures/test_result.png`
    - 全目的変数の要約（`outputs/experiments/x50_{モデル名}_summary/{実行日時}/`）:
      `summary.csv`（目的変数ごとの MAE）、`test_predictions.csv`（全目的変数のテスト予測）、
      `figures/summary_mae.png`・`figures/test_overview.png`・`figures/shap_overview.png`、`run.log`
    - MLflow（`mlruns/`）: 目的変数ごとの run と、要約の run（同じ `batch` タグで結び付ける）

使い方:
    uv run python scripts/train_predict_x50.py --model lightgbm
    uv run python scripts/train_predict_x50.py --model xgboost --targets Y1 Y2 --n-trials 20
    uv run python scripts/train_predict_x50.py --model lightgbm --no-tracking --no-answer
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import re
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from modeling.config import ExperimentConfig  # noqa: E402
from modeling.dataset import Dataset  # noqa: E402
from modeling.experiment import ExperimentResult, prepare_dataset, run_experiment  # noqa: E402
from modeling.forecasting import ForecastCVResult  # noqa: E402
from modeling.metrics import get_metric  # noqa: E402
from modeling.tracking import MLflowTracker, NullTracker, Tracker  # noqa: E402
from util.csv_io import read_csv_auto  # noqa: E402
from util.paths import data_dir, ensure_parent_dir, get_repo_root, outputs_dir  # noqa: E402
from util.plotting import add_caption, ensure_japanese_font  # noqa: E402

MODELS = ("lightgbm", "xgboost")
TRAIN_PATH = data_dir() / "processed" / "x50" / "x50_train_features.csv"
TEST_PATH = data_dir() / "processed" / "x50" / "x50_test_features.csv"
ANSWER_PATH = data_dir() / "synthetic" / "x50" / "dummy_x50_test_answer.csv"

DATE_COL = "date"
RAW_SUFFIX = "_raw"  # 学習データのCSVの、元の尺度の目的変数の列名の接尾辞（`Yk_raw`）
PRED_COL = "pred"  # 目的変数ごとのテスト予測の表の、予測値の列
ACTUAL_COL = "actual"  # 同じく、正解の列（正解ファイルがある場合）

# 検証期間の開始日（時系列CVの各foldの境目）。最後の fold は学習データの最終日まで
DEFAULT_CUTOFFS = ["2023-07-01", "2024-01-01"]
MLFLOW_EXPERIMENT = "x50_forecast"

logger = logging.getLogger("train_predict_x50")


class LeakageError(ValueError):
    """入力データにリークのおそれがある場合のエラー。"""


@dataclass(frozen=True)
class TargetFeatureSpec:
    """学習データのCSVにある、1つの目的変数から作った特徴量の定義（列名から読み取る）。

    Attributes:
        target: 目的変数の列名（例: `Y1`）。
        lags: ラグ（`{target}_lag_{k}`）。
        rolling_windows: 前日までの移動平均の窓幅（`{target}_lag_1_ma_{w}`）。
        rate_of_change: 前日の前日比（`{target}_lag_1_roc_1`）があるか。
        columns: これらの列名（CSVの並び順）。
    """

    target: str
    lags: list[int]
    rolling_windows: list[int]
    rate_of_change: bool
    columns: list[str] = field(default_factory=list)


@dataclass
class Inputs:
    """読み込んで確認した入力。

    Attributes:
        train: 学習データのCSV（日付の昇順）。
        test: テストデータのCSV（日付の昇順）。
        targets: 目的変数の列名（CSVの並び順）。
        specs: 目的変数 → その目的変数の特徴量の定義。
        exog_cols: 日付の特徴量・説明変数の列（学習・テストで共通。全目的変数のモデルで使う）。
    """

    train: pd.DataFrame
    test: pd.DataFrame
    targets: list[str]
    specs: dict[str, TargetFeatureSpec]
    exog_cols: list[str]


@dataclass
class TargetRun:
    """1つの目的変数の学習・予測の結果。

    Attributes:
        target: 目的変数の列名。
        dataset: 学習用データ一式。
        result: `run_experiment` の結果。
        predictions: テスト予測（列 `date`, `pred`。正解があれば `actual` も）。
        test_mae: テストの MAE（正解が無ければNone）。
    """

    target: str
    dataset: Dataset
    result: ExperimentResult
    predictions: pd.DataFrame
    test_mae: float | None


# --- ログ -------------------------------------------------------------------------------


def setup_logging(log_path: Path, level: int = logging.INFO) -> Path:
    """画面とファイルの両方にログを出す設定をする（同じ設定を2回呼んでも重複しない）。

    Args:
        log_path: ログファイルのパス（UTF-8）。
        level: ログの出力レベル。

    Returns:
        ログファイルのパス。
    """
    ensure_parent_dir(log_path)
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        if getattr(handler, "_x50", False):
            root.removeHandler(handler)
            handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers: list[logging.Handler] = [
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
    for handler in handlers:
        handler.setFormatter(formatter)
        handler._x50 = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    # ライブラリの警告もログに残す（細かい情報ログは出さない）
    logging.captureWarnings(True)
    for name in ("matplotlib", "PIL", "shap", "urllib3", "alembic", "mlflow"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return log_path


def shown(path: Path) -> str:
    """ログ・画面に出すパス（リポジトリ内ならルートからの相対パス。個人のフォルダ名を残さない）。"""
    resolved = Path(path).resolve()
    root = get_repo_root()
    return resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else str(path)


def file_digest(path: Path) -> str:
    """ファイルの SHA-256（先頭12文字）。どのデータで学習したかをログに残すために使う。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


# --- 読み込みとリークの確認 -------------------------------------------------------------


def detect_targets(columns: Sequence[str]) -> list[str]:
    """学習データの列から目的変数を見つける（`Yk` と `Yk_raw` の両方がある列 `Yk`）。"""
    return [c for c in columns if f"{c}{RAW_SUFFIX}" in columns]


def parse_target_features(columns: Sequence[str], target: str) -> TargetFeatureSpec:
    """列名から、1つの目的変数のラグ・移動平均・前日比の定義を読み取る。

    Raises:
        ValueError: 再帰予測で作り直せない目的変数の特徴量がある場合
            （2日以上前のラグの移動平均など）、またはラグが無い場合。
    """
    lags, windows, roc, found = [], [], False, []
    prefix = f"{target}_lag_"
    for col in columns:
        if not col.startswith(prefix):
            continue
        found.append(col)
        if m := re.fullmatch(rf"{re.escape(prefix)}(\d+)", col):
            lags.append(int(m.group(1)))
        elif m := re.fullmatch(rf"{re.escape(prefix)}1_ma_(\d+)", col):
            windows.append(int(m.group(1)))
        elif col == f"{prefix}1_roc_1":
            roc = True
        else:
            raise ValueError(f"再帰予測で作り直せない目的変数の特徴量です: {col}")
    if not lags:
        raise ValueError(f"学習データに目的変数のラグ（{prefix}k）がありません")
    return TargetFeatureSpec(target, sorted(lags), sorted(windows), roc, found)


def check_daily(frame: pd.DataFrame, name: str) -> None:
    """日付が1日刻みで、重複も欠けもないことを確かめる（ラグは1行=1日の前提で作るため）。"""
    gaps = frame[DATE_COL].diff().dropna()
    if frame[DATE_COL].duplicated().any() or not (gaps == pd.Timedelta(days=1)).all():
        raise ValueError(f"{name}の日付が1日刻みになっていません（重複または欠け）")


def load_inputs(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    targets: Sequence[str] | None = None,
) -> Inputs:
    """前処理済みの学習・テストデータを読み込み、リークのおそれがないことを確かめる。

    確かめること:
        - テストデータに、どの目的変数も・目的変数から作った特徴量も無い。
        - 学習・テストで、日付の特徴量・説明変数の列が同じ。
        - どちらも1日刻みで、テストデータは学習データの最終日の翌日から始まる。

    Args:
        train_path: 学習データのCSV。
        test_path: テストデータのCSV。
        targets: 予測する目的変数（Noneなら学習データの全目的変数）。

    Raises:
        LeakageError: テストデータに目的変数（またはその特徴量）がある場合。
        ValueError: 列や日付が前提を満たさない場合。
    """
    train = read_csv_auto(train_path).sort_values(DATE_COL, kind="stable").reset_index(drop=True)
    test = read_csv_auto(test_path).sort_values(DATE_COL, kind="stable").reset_index(drop=True)
    for name, path, frame in (("学習", train_path, train), ("テスト", test_path, test)):
        logger.info(
            "%sデータ: %s（%d行 × %d列, sha256=%s）",
            name,
            path.name,
            *frame.shape,
            file_digest(path),
        )

    all_targets = detect_targets(list(train.columns))
    if not all_targets:
        raise ValueError(f"学習データに目的変数（Yk と Yk{RAW_SUFFIX} の組）がありません")
    leaked = [
        c
        for c in test.columns
        if any(c == t or c == f"{t}{RAW_SUFFIX}" or c.startswith(f"{t}_") for t in all_targets)
    ]
    if leaked:
        raise LeakageError(f"テストデータに目的変数（またはその特徴量）があります: {leaked}")
    unknown = [t for t in targets or [] if t not in all_targets]
    if unknown:
        raise ValueError(f"学習データに無い目的変数です: {unknown}（あるのは {all_targets}）")

    columns = list(train.columns)
    specs = {t: parse_target_features(columns, t) for t in all_targets}
    target_cols = {c for t in all_targets for c in (t, f"{t}{RAW_SUFFIX}", *specs[t].columns)}
    exog_cols = [c for c in columns if c != DATE_COL and c not in target_cols]
    if set(exog_cols) != set(test.columns) - {DATE_COL}:
        only_train = sorted(set(exog_cols) - set(test.columns))
        only_test = sorted(set(test.columns) - set(exog_cols) - {DATE_COL})
        raise ValueError(
            f"学習・テストで特徴量の列が違います（学習のみ: {only_train}, テストのみ: {only_test}）"
        )

    check_daily(train, "学習データ")
    check_daily(test, "テストデータ")
    train_end, test_start = train[DATE_COL].max(), test[DATE_COL].min()
    if test_start != train_end + pd.Timedelta(days=1):
        raise ValueError(
            f"テストデータが学習データの翌日から始まっていません: {train_end} → {test_start}"
        )
    selected = list(targets) if targets else all_targets
    first = specs[selected[0]]
    logger.info(
        "期間: 学習 %s〜%s / テスト %s〜%s",
        f"{train[DATE_COL].min():%Y-%m-%d}",
        f"{train_end:%Y-%m-%d}",
        f"{test_start:%Y-%m-%d}",
        f"{test[DATE_COL].max():%Y-%m-%d}",
    )
    logger.info("目的変数: %s", selected)
    logger.info(
        "特徴量（目的変数ごと）: 自身のラグ等 %d列（ラグ %s, 移動平均 %s, 前日比 %s。"
        "再帰予測で作り直す）, 日付・説明変数 %d列",
        len(first.columns),
        first.lags,
        first.rolling_windows,
        first.rate_of_change,
        len(exog_cols),
    )
    return Inputs(
        train=train,
        test=test[[DATE_COL, *exog_cols]],
        targets=selected,
        specs={t: specs[t] for t in selected},
        exog_cols=exog_cols,
    )


def model_frames(inputs: Inputs, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1つの目的変数について、`prepare_dataset` に渡す学習・テストの表を作る。

    学習の表は「日付・日付の特徴量・説明変数・目的変数（元の尺度）」だけにする。
    他の目的変数の列と、CSVにある目的変数の特徴量は外す（目的変数の特徴量は `modeling` に
    同じ定義で作り直させる。テスト期間は再帰予測の中で作るため）。
    """
    train = inputs.train[[DATE_COL, *inputs.exog_cols]].assign(
        **{target: inputs.train[f"{target}{RAW_SUFFIX}"]}
    )
    test = inputs.test[[DATE_COL, *inputs.exog_cols]]
    assert target not in test.columns
    return train, test


# --- 実験設定 ---------------------------------------------------------------------------


def make_config(
    model: str,
    spec: TargetFeatureSpec,
    cutoffs: Sequence[str] = DEFAULT_CUTOFFS,
    n_trials: int = 0,
    test_prediction: str = "refit_full",
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
) -> ExperimentConfig:
    """1つの目的変数・1モデル分の実験設定を作る（`ExperimentConfig` で検証する）。

    Args:
        model: モデル名（`lightgbm` / `xgboost`）。
        spec: 目的変数とその特徴量の定義（学習データのCSVと同じものを再帰予測でも作る）。
        cutoffs: 時系列CVの検証期間の開始日。
        n_trials: Optunaの試行回数（0ならチューニングしない）。
        test_prediction: テスト予測の作り方（`refit_full` は学習データ全体で学習し直したモデル、
            `fold_mean` は各foldのモデルの平均）。
        train_path: 記録用の学習データのパス。
        test_path: 記録用のテストデータのパス。

    Returns:
        検証済みの実験設定。

    Raises:
        ValueError: 未対応のモデル名の場合。
    """
    if model not in MODELS:
        raise ValueError(f"モデルは {MODELS} から選んでください: {model}")
    return ExperimentConfig.model_validate(
        {
            "name": f"x50_{model}_{spec.target}",
            # time_series では時間順を無視したCV（kfold 等）は設定エラーになる
            "task": "time_series",
            "data": {
                "train_path": str(train_path),
                "test_path": str(test_path),
                "target": spec.target,
                # 時刻列。学習データはこの列の昇順に並べ替えられる。モデルの特徴量からは外す
                "time_col": DATE_COL,
                "drop_cols": [DATE_COL],
            },
            # 特徴量の前処理は 003 のノートブックで学習データだけを使って済ませているため、
            # ここでは行わない
            "features": [],
            # 日付で区切る時系列CV。各 fold は「境目より前で学習し、境目以降を再帰予測して評価」する
            "cv": {"method": "time_cutoff", "time_column": DATE_COL, "cutoffs": list(cutoffs)},
            "model": {"name": model, "params": {"n_estimators": 1000}, "early_stopping_rounds": 50},
            "metrics": ["mae"],
            "test_prediction": test_prediction,
            "forecast": {
                "lags": spec.lags,
                "rolling_windows": spec.rolling_windows,
                "rate_of_change": spec.rate_of_change,
                # 目的変数は対数で学習・予測し、評価・出力の前に元の尺度に戻す
                # （003 の前処理と同じ変換）
                "target_transform": "log",
            },
            "tuning": {"enabled": n_trials > 0, "n_trials": max(n_trials, 1)},
            "explain": {"enabled": True},
            "evaluation": {"enabled": True, "training_history": True},
            "tracking": {"experiment_name": MLFLOW_EXPERIMENT},
        }
    )


# --- 学習と確認 -------------------------------------------------------------------------


def prepare(config: ExperimentConfig, inputs: Inputs) -> Dataset:
    """1つの目的変数の学習用データ一式を作り、リークのおそれがないことを確かめる。

    確かめること:
        - 作り直した目的変数の特徴量が、学習データのCSVの値と一致する（CSVの特徴量も過去の値だけから
          作られている、かつ定義が同じ）。作り直せない先頭の行（過去の日数が足りない）は欠損になる。
        - モデルの特徴量に、他の目的変数（やその特徴量）が入っていない。
        - CVの各foldで、学習期間がすべて検証期間より前にある。
        - テストデータに目的変数が無い（再帰予測は予測値だけを履歴に足す）。

    Raises:
        LeakageError: 上のいずれかを満たさない場合。
    """
    target = config.data.target
    spec = inputs.specs[target]
    train, test = model_frames(inputs, target)
    dataset = prepare_dataset(config, train, test)

    rebuilt = dataset.X[spec.columns].reset_index(drop=True)
    original = inputs.train[spec.columns]
    comparable = rebuilt.notna()
    if not np.allclose(
        rebuilt[comparable], original[comparable], rtol=1e-9, atol=1e-9, equal_nan=True
    ):
        raise LeakageError(
            f"{target}: 作り直した目的変数の特徴量が、学習データのCSVの値と一致しません"
        )
    others = [t for t in inputs.specs if t != target]
    mixed = [c for c in dataset.X.columns if any(c == t or c.startswith(f"{t}_") for t in others)]
    if mixed:
        raise LeakageError(f"{target}: モデルの特徴量に他の目的変数があります: {mixed}")
    n_head = int((~comparable.all(axis=1)).sum())
    logger.info(
        "%s: 目的変数の特徴量はCSVの値と一致（先頭 %d行は過去の日数が足りないため欠損）",
        target,
        n_head,
    )

    dates = pd.to_datetime(train[DATE_COL])
    for i, (train_idx, valid_idx) in enumerate(dataset.folds):
        if dates.iloc[train_idx].max() >= dates.iloc[valid_idx].min():
            raise LeakageError(f"{target} fold {i}: 学習期間が検証期間より後の日を含みます")
        logger.debug(
            "%s fold %d: 学習 %s〜%s → 検証 %s〜%s",
            target,
            i,
            f"{dates.iloc[train_idx].min():%Y-%m-%d}",
            f"{dates.iloc[train_idx].max():%Y-%m-%d}",
            f"{dates.iloc[valid_idx].min():%Y-%m-%d}",
            f"{dates.iloc[valid_idx].max():%Y-%m-%d}",
        )
    if dataset.test_frame is None or target in dataset.test_frame.columns:
        raise LeakageError(
            f"{target}: テストデータに目的変数があります（またはテストデータがありません）"
        )
    return dataset


def log_folds(dataset: Dataset) -> None:
    """CVの各foldの期間をログに出す（全目的変数で同じ分割なので1回だけ出す）。"""
    assert dataset.frame is not None
    dates = pd.to_datetime(dataset.frame[DATE_COL])
    for i, (train_idx, valid_idx) in enumerate(dataset.folds):
        logger.info(
            "fold %d: 学習 %s〜%s（%d日）→ 検証 %s〜%s（%d日、再帰予測で評価）",
            i,
            f"{dates.iloc[train_idx].min():%Y-%m-%d}",
            f"{dates.iloc[train_idx].max():%Y-%m-%d}",
            len(train_idx),
            f"{dates.iloc[valid_idx].min():%Y-%m-%d}",
            f"{dates.iloc[valid_idx].max():%Y-%m-%d}",
            len(valid_idx),
        )


def make_tracker(enabled: bool) -> Tracker:
    """MLflow に記録する（`enabled=False` なら何も記録しない）。"""
    return MLflowTracker(MLFLOW_EXPERIMENT) if enabled else NullTracker()


def train_and_predict(
    config: ExperimentConfig,
    dataset: Dataset,
    tracker: Tracker,
    output_root: Path | None = None,
    extra_tags: dict[str, str] | None = None,
) -> ExperimentResult:
    """CV学習（再帰バックテスト）・テスト予測・誤差評価・SHAP を `run_experiment` で行う。"""
    target = config.data.target
    start = time.perf_counter()
    result = run_experiment(
        config,
        dataset,
        tracker=tracker,
        output_root=output_root or outputs_dir() / "experiments",
        extra_tags={"target": target, **(extra_tags or {})},
    )
    cv = result.cv_result
    onestep = (
        f" / 参考: 1日先予測 {cv.onestep_oof_scores['mae']:.4f}"
        if isinstance(cv, ForecastCVResult)
        else ""
    )
    logger.info(
        "%s: CV MAE（再帰予測）fold別 %s / OOF %.4f%s（%.1f秒）",
        target,
        [round(s, 4) for s in cv.fold_scores["mae"]],
        cv.oof_scores["mae"],
        onestep,
        time.perf_counter() - start,
    )
    if result.tuning_result is not None:
        logger.info(
            "%s: チューニングの最良パラメータ: %s", target, result.tuning_result.best_params
        )
    return result


def test_predictions(dataset: Dataset, result: ExperimentResult) -> pd.DataFrame:
    """テスト予測を日付つきの表にする（元の尺度）。"""
    assert dataset.test_frame is not None and result.cv_result.test_pred is not None
    return pd.DataFrame(
        {
            DATE_COL: dataset.test_frame[DATE_COL].to_numpy(),
            PRED_COL: result.cv_result.test_pred,
        }
    )


def load_answer(answer_path: Path) -> pd.DataFrame:
    """テストの正解を読む（テスト予測が終わった後の評価にだけ使う）。"""
    answer = read_csv_auto(answer_path)
    logger.info("正解: %s（%d行。評価にだけ使う）", answer_path.name, len(answer))
    return answer


def evaluate_test(
    predictions: pd.DataFrame, answer: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, float]:
    """テスト予測を正解と比べ、MAE を返す。

    Returns:
        (正解の列 `actual` を付けたテスト予測, MAE)。
    """
    merged = predictions.merge(
        answer[[DATE_COL, target]].rename(columns={target: ACTUAL_COL}),
        on=DATE_COL,
        how="left",
        validate="one_to_one",
    )
    if merged[ACTUAL_COL].isna().any():
        raise ValueError(f"正解ファイルに、{target} のテスト期間の一部の日付がありません")
    mae = get_metric("mae")(merged[ACTUAL_COL].to_numpy(), merged[PRED_COL].to_numpy())
    logger.info("%s: テスト MAE %.4f（%d日）", target, mae, len(merged))
    return merged, mae


# --- 図（目的変数ごと） -------------------------------------------------------------------


def plot_training_result(dataset: Dataset, result: ExperimentResult, title: str) -> plt.Figure:
    """学習結果: 学習期間の実測値と、各 fold の検証期間の再帰予測（OOF）を重ねて描く。"""
    ensure_japanese_font()
    assert dataset.frame is not None
    cv = result.cv_result
    dates = pd.to_datetime(dataset.frame[DATE_COL]).to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True, height_ratios=[3, 2])
    ax = axes[0]
    ax.plot(dates, dataset.y, color="black", linewidth=1, label="実測")
    mae = get_metric("mae")
    for fold in range(len(cv.fold_scores["mae"])):
        mask = cv.fold_ids == fold
        if not mask.any():
            continue
        ax.axvspan(dates[mask].min(), dates[mask].max(), color=f"C{fold}", alpha=0.08)
        ax.plot(
            dates[mask],
            cv.oof_pred[mask],
            color=f"C{fold}",
            linewidth=1.2,
            label=f"fold {fold} の再帰予測（MAE={mae(dataset.y[mask], cv.oof_pred[mask]):.2f}）",
        )
    ax.set_title(f"{title}: 学習期間の実測値と、検証期間の再帰予測")
    ax.set_ylabel(dataset_target(dataset))
    ax.legend(loc="upper left")

    # 検証期間の残差（実測 − 予測）。予測が先に進むほど誤差が広がるかを見る
    ax = axes[1]
    predicted = cv.fold_ids >= 0
    residual = dataset.y - cv.oof_pred
    for fold in np.unique(cv.fold_ids[predicted]):
        mask = cv.fold_ids == fold
        ax.plot(dates[mask], residual[mask], color=f"C{fold}", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("残差（実測 − 予測）")
    ax.set_xlabel("日付")
    ax.set_title("検証期間の残差")
    add_caption(
        fig,
        f"OOF MAE={cv.oof_scores['mae']:.3f}。網掛けは各 fold の検証期間。"
        "検証期間は実測値を使わず、予測値を次のラグに使う再帰予測で評価した。",
    )
    return fig


def dataset_target(dataset: Dataset) -> str:
    """学習用データ一式の目的変数名（図の軸ラベル用）。"""
    assert dataset.frame is not None
    names = [c for c in dataset.frame.columns if c not in dataset.X.columns and c != DATE_COL]
    return names[0] if names else "目的変数"


def plot_test_result(
    dataset: Dataset, predictions: pd.DataFrame, title: str, history_days: int = 180
) -> plt.Figure:
    """テスト結果: テスト期間の予測（正解があれば実測と比較）を描く。

    正解（列 `actual`）があれば、時系列・予測と実測の散布図・月別MAEの3つを描く。
    無ければ時系列だけ。
    """
    ensure_japanese_font()
    assert dataset.frame is not None
    has_answer = ACTUAL_COL in predictions.columns
    if has_answer:
        fig, axes = plt.subplot_mosaic(
            [["series", "series"], ["scatter", "monthly"]],
            figsize=(13, 9),
            constrained_layout=True,
            height_ratios=[3, 2],
        )
    else:
        fig, ax = plt.subplots(figsize=(13, 5), constrained_layout=True)
        axes = {"series": ax}
    history = dataset.frame.tail(history_days)
    ax = axes["series"]
    ax.plot(
        history[DATE_COL],
        dataset.y[-history_days:],
        color="gray",
        linewidth=1,
        label="学習期間の実測（直近）",
    )
    if has_answer:
        ax.plot(
            predictions[DATE_COL],
            predictions[ACTUAL_COL],
            color="black",
            linewidth=1,
            label="テスト期間の実測",
        )
    ax.plot(
        predictions[DATE_COL], predictions[PRED_COL], color="C3", linewidth=1.5, label="再帰予測"
    )
    ax.axvline(predictions[DATE_COL].min(), color="gray", linestyle="--", linewidth=0.8)
    period = f"{predictions[DATE_COL].min():%Y-%m-%d}〜{predictions[DATE_COL].max():%Y-%m-%d}"
    ax.set_title(f"{title}: テスト期間（{period}）の予測")
    ax.set_ylabel(dataset_target(dataset))
    ax.legend(loc="upper left")

    caption = "学習データの最終日の翌日から、予測値を次のラグに使う再帰予測で1日ずつ予測した。"
    if has_answer:
        actual, pred = predictions[ACTUAL_COL], predictions[PRED_COL]
        mae = float(np.mean(np.abs(actual - pred)))
        ax = axes["scatter"]
        ax.scatter(actual, pred, s=10, alpha=0.6)
        low, high = float(min(actual.min(), pred.min())), float(max(actual.max(), pred.max()))
        ax.plot([low, high], [low, high], color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("実測")
        ax.set_ylabel("予測")
        ax.set_title(f"予測と実測（MAE={mae:.3f}）")

        # 月ごとのMAE（予測が先に進むほど誤差が大きくなるか）
        monthly = (actual - pred).abs().groupby(predictions[DATE_COL].dt.to_period("M")).mean()
        ax = axes["monthly"]
        positions = np.arange(len(monthly))
        ax.bar(positions, monthly.to_numpy(), color="C0")
        ax.set_xticks(positions, [str(p) for p in monthly.index])
        ax.set_ylim(bottom=0)  # 棒グラフは0起点
        ax.set_ylabel("MAE")
        ax.set_title("月別のMAE")
        ax.tick_params(axis="x", rotation=60)
        caption += f" テスト MAE={mae:.3f}（正解は予測後の評価にだけ使用）。"
    add_caption(fig, caption)
    return fig


def save_figure(fig: plt.Figure, path: Path) -> Path:
    """図を保存して閉じる。"""
    fig.savefig(ensure_parent_dir(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("図を保存: %s", shown(path))
    return path


def save_target_outputs(run: TargetRun) -> list[Path]:
    """1つの目的変数のテスト予測・スコア・図を実験の出力ディレクトリに保存し、MLflowにも記録する。"""
    out = run.result.output_dir
    title = out.parent.name
    paths = [ensure_parent_dir(out / "test_predictions.csv")]
    run.predictions.to_csv(paths[0], index=False, date_format="%Y-%m-%d")
    if run.test_mae is not None:
        paths.append(out / "test_scores.csv")
        scores = pd.DataFrame(
            {"metric": ["mae"], "value": [run.test_mae], "n": [len(run.predictions)]}
        )
        scores.to_csv(paths[-1], index=False)
    figures = out / "figures"
    training = plot_training_result(run.dataset, run.result, title)
    paths.append(save_figure(training, figures / "training_result.png"))
    test = plot_test_result(run.dataset, run.predictions, title)
    paths.append(save_figure(test, figures / "test_result.png"))
    # run_experiment のrun（終了済み）に、テストの指標と追加の出力を記録する
    if run.result.run_id is not None:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        if run.test_mae is not None:
            client.log_metric(run.result.run_id, "test_mae", run.test_mae)
        for path in paths:
            artifact_path = "test" if path.suffix == ".csv" else "figures"
            client.log_artifact(run.result.run_id, str(path), artifact_path=artifact_path)
    return paths


def run_target(
    model: str,
    target: str,
    inputs: Inputs,
    tracker: Tracker,
    answer: pd.DataFrame | None = None,
    cutoffs: Sequence[str] = DEFAULT_CUTOFFS,
    n_trials: int = 0,
    test_prediction: str = "refit_full",
    output_root: Path | None = None,
    batch: str | None = None,
) -> TargetRun:
    """1つの目的変数について、設定・確認・学習・テスト予測・評価・出力の保存までを行う。

    Args:
        model: モデル名。
        target: 目的変数。
        inputs: `load_inputs` の結果。
        tracker: 実験ログの記録先。
        answer: テストの正解（Noneなら評価しない）。予測が終わった後にだけ使う。
        cutoffs: 時系列CVの検証期間の開始日。
        n_trials: Optunaの試行回数（0ならチューニングしない）。
        test_prediction: テスト予測の作り方。
        output_root: 出力のルート（Noneなら `outputs/experiments`）。
        batch: 同じ実行の run を結び付ける MLflow のタグ。

    Returns:
        結果。
    """
    config = make_config(model, inputs.specs[target], cutoffs, n_trials, test_prediction)
    dataset = prepare(config, inputs)
    result = train_and_predict(
        config, dataset, tracker, output_root, extra_tags={"batch": batch} if batch else None
    )
    predictions = test_predictions(dataset, result)
    test_mae = None
    if answer is not None:
        predictions, test_mae = evaluate_test(predictions, answer, target)
    run = TargetRun(target, dataset, result, predictions, test_mae)
    save_target_outputs(run)
    return run


# --- 全目的変数の要約 -------------------------------------------------------------------


def summary_frame(runs: Sequence[TargetRun]) -> pd.DataFrame:
    """目的変数ごとの MAE をまとめた表。

    目的変数ごとに尺度が違うため、MAE を実測値の平均で割った相対MAE も載せる
    （CV は検証期間の実測、テストはテスト期間の実測の平均で割る）。
    """
    rows = []
    for run in runs:
        cv = run.result.cv_result
        predicted = cv.fold_ids >= 0
        row: dict[str, object] = {
            "target": run.target,
            "cv_oof_mae": cv.oof_scores["mae"],
            "cv_relative_mae": cv.oof_scores["mae"] / float(np.mean(run.dataset.y[predicted])),
        }
        if isinstance(cv, ForecastCVResult):
            row["cv_onestep_mae"] = cv.onestep_oof_scores["mae"]
        if run.test_mae is not None:
            row["test_mae"] = run.test_mae
            row["test_relative_mae"] = run.test_mae / float(run.predictions[ACTUAL_COL].mean())
        row["output_dir"] = shown(run.result.output_dir)
        row["run_id"] = run.result.run_id
        rows.append(row)
    return pd.DataFrame(rows)


def wide_predictions(runs: Sequence[TargetRun]) -> pd.DataFrame:
    """全目的変数のテスト予測を横に並べた表（列 `date`, `Yk_pred`, 正解があれば `Yk_actual`）。"""
    frame = runs[0].predictions[[DATE_COL]].copy()
    for run in runs:
        frame[f"{run.target}_pred"] = run.predictions[PRED_COL].to_numpy()
        if ACTUAL_COL in run.predictions.columns:
            frame[f"{run.target}_actual"] = run.predictions[ACTUAL_COL].to_numpy()
    return frame


def plot_summary_mae(summary: pd.DataFrame, title: str) -> plt.Figure:
    """目的変数ごとの相対MAE（CV とテスト）の棒グラフ。"""
    ensure_japanese_font()
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    positions = np.arange(len(summary))
    has_test = "test_relative_mae" in summary.columns
    width = 0.4 if has_test else 0.6
    ax.bar(
        positions - (width / 2 if has_test else 0),
        summary["cv_relative_mae"],
        width,
        label="CV（検証期間の再帰予測）",
    )
    if has_test:
        ax.bar(positions + width / 2, summary["test_relative_mae"], width, label="テスト（2025年）")
    ax.set_xticks(positions, summary["target"])
    ax.set_ylim(bottom=0)  # 棒グラフは0起点
    ax.set_ylabel("相対MAE（MAE ÷ 実測値の平均）")
    ax.set_title(f"{title}: 目的変数ごとの誤差")
    ax.legend(loc="upper left")
    add_caption(
        fig,
        "目的変数ごとに水準が違うため、MAE を実測値の平均で割って比べる。"
        "CV とテストの差が大きい目的変数は、期間によって当てやすさが違う（過学習・トレンドなど）。",
    )
    return fig


def plot_test_overview(runs: Sequence[TargetRun], title: str, ncols: int = 2) -> plt.Figure:
    """全目的変数のテスト期間の予測（正解があれば実測も）を並べて描く。"""
    ensure_japanese_font()
    nrows = int(np.ceil(len(runs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 2.6 * nrows), constrained_layout=True, sharex=True, squeeze=False
    )
    for ax, run in zip(axes.flat, runs, strict=False):
        p = run.predictions
        if ACTUAL_COL in p.columns:
            ax.plot(p[DATE_COL], p[ACTUAL_COL], color="black", linewidth=0.8, label="実測")
        ax.plot(p[DATE_COL], p[PRED_COL], color="C3", linewidth=1.2, label="再帰予測")
        label = run.target if run.test_mae is None else f"{run.target}（MAE={run.test_mae:.2f}）"
        ax.set_title(label)
    for ax in list(axes.flat)[len(runs) :]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(f"{title}: テスト期間の予測（目的変数ごと）")
    return fig


def shap_importance_table(runs: Sequence[TargetRun]) -> pd.DataFrame:
    """目的変数ごとの SHAP 重要度（平均|SHAP値|の割合）を、特徴量 × 目的変数の表にする。

    目的変数自身のラグ等は、目的変数によらず同じ行に並ぶよう `Yk_lag_1` → `自身_lag_1` に
    名前を揃える。
    重要度は目的変数ごとに合計1になるよう割合にする（目的変数によって尺度が違うため）。
    """
    columns = {}
    for run in runs:
        path = run.result.output_dir / "shap" / "shap_importance.csv"
        importance = pd.read_csv(path).set_index("feature")["mean_abs_shap"]
        importance.index = [
            f"自身{name[len(run.target) :]}" if name.startswith(f"{run.target}_") else name
            for name in importance.index
        ]
        columns[run.target] = importance / importance.sum()
    return pd.DataFrame(columns).fillna(0.0)


def plot_shap_overview(table: pd.DataFrame, title: str, top_k: int = 15) -> plt.Figure:
    """SHAP 重要度の割合を、上位の特徴量 × 目的変数のヒートマップで描く。"""
    ensure_japanese_font()
    top = table.loc[table.mean(axis=1).sort_values(ascending=False).index[:top_k]]
    fig, ax = plt.subplots(
        figsize=(1.0 * top.shape[1] + 4, 0.4 * top_k + 2), constrained_layout=True
    )
    image = ax.imshow(top.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(top.shape[1]), top.columns)
    ax.set_yticks(range(top.shape[0]), top.index)
    for i in range(top.shape[0]):
        for j in range(top.shape[1]):
            value = top.iat[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value < top.to_numpy().max() * 0.6 else "black",
            )
    fig.colorbar(image, ax=ax, label="重要度の割合（平均|SHAP値|。目的変数ごとに合計1）")
    ax.set_title(f"{title}: 目的変数ごとの SHAP 重要度（上位{top_k}特徴量）")
    add_caption(fig, "「自身_lag_1」などは、各目的変数自身のラグ・移動平均・前日比。")
    return fig


def save_summary(
    runs: Sequence[TargetRun],
    model: str,
    summary_dir: Path,
    tracker: Tracker,
    log_path: Path | None = None,
    batch: str | None = None,
) -> pd.DataFrame:
    """全目的変数の要約（表・図・ログ）を保存し、MLflow に要約の run として記録する。"""
    title = f"x50_{model}"
    summary = summary_frame(runs)
    paths = [ensure_parent_dir(summary_dir / "summary.csv")]
    summary.to_csv(paths[0], index=False)
    paths.append(summary_dir / "test_predictions.csv")
    wide_predictions(runs).to_csv(paths[-1], index=False, date_format="%Y-%m-%d")
    paths.append(summary_dir / "shap_importance_by_target.csv")
    shap_table = shap_importance_table(runs)
    shap_table.to_csv(paths[-1], index_label="feature")
    figures = summary_dir / "figures"
    paths.append(save_figure(plot_summary_mae(summary, title), figures / "summary_mae.png"))
    paths.append(save_figure(plot_test_overview(runs, title), figures / "test_overview.png"))
    paths.append(save_figure(plot_shap_overview(shap_table, title), figures / "shap_overview.png"))

    mean_cv = float(summary["cv_relative_mae"].mean())
    message = f"要約: CV 相対MAE の平均 {mean_cv:.4f}"
    if "test_relative_mae" in summary.columns:
        message += f" / テスト 相対MAE の平均 {float(summary['test_relative_mae'].mean()):.4f}"
    logger.info("%s（%d目的変数）。表: %s", message, len(runs), shown(paths[0]))
    if log_path is not None:
        for handler in logging.getLogger().handlers:
            handler.flush()
        paths.append(Path(shutil.copy(log_path, summary_dir / "run.log")))

    tags = {"type": "summary", "model": model, **({"batch": batch} if batch else {})}
    with tracker.start_run(run_name=f"{title}_summary", tags=tags):
        tracker.log_params({"model": model, "targets": [r.target for r in runs]})
        metrics = {"mean_cv_relative_mae": mean_cv}
        for row in summary.to_dict("records"):
            metrics[f"cv_oof_mae_{row['target']}"] = row["cv_oof_mae"]
            if "test_mae" in row:
                metrics[f"test_mae_{row['target']}"] = row["test_mae"]
        if "test_relative_mae" in summary.columns:
            metrics["mean_test_relative_mae"] = float(summary["test_relative_mae"].mean())
        tracker.log_metrics(metrics)
        for path in paths:
            tracker.log_artifact(path)
        run_id = tracker.active_run_id()
    if run_id is not None:
        logger.info("MLflow に要約を記録: run_id=%s（batch=%s）", run_id, batch)
    return summary


# --- メイン処理 -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """全目的変数の学習・テスト予測・評価・図・ログの保存を行う。"""
    import matplotlib

    # 図はファイルに保存するだけなので、画面を使わないバックエンドにする
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODELS, default="lightgbm", help="モデル")
    parser.add_argument("--targets", nargs="+", default=None, help="目的変数（既定: 全目的変数）")
    parser.add_argument(
        "--n-trials", type=int, default=0, help="Optunaの試行回数（0ならチューニングしない）"
    )
    parser.add_argument(
        "--cutoffs", nargs="+", default=DEFAULT_CUTOFFS, help="CVの検証期間の開始日"
    )
    parser.add_argument(
        "--test-prediction", choices=["refit_full", "fold_mean"], default="refit_full"
    )
    parser.add_argument("--train", type=Path, default=TRAIN_PATH, help="学習データのCSV")
    parser.add_argument("--test", type=Path, default=TEST_PATH, help="テストデータのCSV")
    parser.add_argument(
        "--answer", type=Path, default=ANSWER_PATH, help="テストの正解のCSV（評価用）"
    )
    parser.add_argument("--no-answer", action="store_true", help="正解との比較をしない")
    parser.add_argument("--no-tracking", action="store_true", help="MLflowに記録しない")
    parser.add_argument(
        "--output-root", type=Path, default=None, help="出力のルート（既定: outputs/experiments）"
    )
    args = parser.parse_args(argv)

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(outputs_dir() / "logs" / f"train_predict_x50_{args.model}_{stamp}.log")
    logger.info(
        "引数: %s", {k: shown(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    )

    inputs = load_inputs(args.train, args.test, args.targets)
    tracker = make_tracker(not args.no_tracking)
    answer = None if args.no_answer or not args.answer.is_file() else load_answer(args.answer)
    batch = f"x50_{args.model}_{stamp}"
    runs = []
    for i, target in enumerate(inputs.targets):
        logger.info(
            "=== %s（%d/%d）: モデル=%s ===", target, i + 1, len(inputs.targets), args.model
        )
        run = run_target(
            args.model,
            target,
            inputs,
            tracker,
            answer=answer,
            cutoffs=args.cutoffs,
            n_trials=args.n_trials,
            test_prediction=args.test_prediction,
            output_root=args.output_root,
            batch=batch,
        )
        if i == 0:
            log_folds(run.dataset)
        runs.append(run)

    root = args.output_root or outputs_dir() / "experiments"
    summary_dir = root / f"x50_{args.model}_summary" / stamp
    save_summary(runs, args.model, summary_dir, tracker, log_path, batch)
    logger.info("完了: 要約 %s", shown(summary_dir))


if __name__ == "__main__":
    main()
