"""特徴量エンジニアリングの動作確認用に、日次のダミーデータ（date, Y1〜Y10, X1〜X50）を作る。

1本の系列として作り、学習期間とテスト期間に分けて保存する（いずれも gitの管理外）。
実データではないため `data/raw` には置かず、合成データ用の `data/synthetic/x50/` に置く。

| ファイル | 期間（既定） | 列 |
|---|---|---|
| `dummy_x50_train.csv` | 2022-01-01〜2024-12-31 | `date`, `Y1`〜`Y10`, `X1`〜`X50` |
| `dummy_x50_test.csv` | 2025-01-01〜2025-12-31 | `date`, `X1`〜`X50`（目的変数なし） |
| `dummy_x50_test_answer.csv` | テストと同じ | `date`, `Y1`〜`Y10`（予測の評価だけに使う正解） |

説明変数は前処理の違いが分かるよう、性質の異なる5種類を10列ずつ用意する。

| 列 | 性質 |
|---|---|
| X1〜X10 | 季節性・トレンドのある連続値（正規分布に近い） |
| X11〜X20 | 右に裾の長い正の値（対数正規分布）。べき変換の対象になる |
| X21〜X30 | 連続値で、ところどころ欠損がある（約2%） |
| X31〜X40 | 件数（ポアソン分布の整数） |
| X41〜X50 | 目的変数と無関係なノイズ |

目的変数 Y1〜Y10 は、前日の自分自身（自己相関）・年周期の季節性・一部の説明変数から作る正の値。
モデルや前処理の違いが見えるよう、目的変数ごとに次を変える。

- 水準（10〜200）と、季節性の強さ・位相
- 自己相関の強さ（前日の値への回帰係数 0.3〜0.9）
- 効く説明変数: Yk には Xk・X(10+k)・X(20+k)・X(30+k) が効く
  （係数の符号・大きさも目的変数ごとに違う）
- Y3・Y6・Y9 には緩やかな上昇トレンドがある

使い方:
    uv run python scripts/generate_dummy_x50.py
    uv run python scripts/generate_dummy_x50.py --test-start 2025-01-01 --end 2025-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from util.paths import data_dir, ensure_parent_dir  # noqa: E402

N_FEATURES = 50
N_TARGETS = 10
TARGET_COLS = [f"Y{k}" for k in range(1, N_TARGETS + 1)]
OUTPUT_DIR = data_dir() / "synthetic" / "x50"
TRAIN_FILENAME = "dummy_x50_train.csv"
TEST_FILENAME = "dummy_x50_test.csv"
ANSWER_FILENAME = "dummy_x50_test_answer.csv"


def make_dummy_frame(start: str = "2022-01-01", days: int = 1096, seed: int = 0) -> pd.DataFrame:
    """ダミーデータを作る。

    Args:
        start: 開始日。
        days: 日数（1行 = 1日）。
        seed: 乱数のシード。

    Returns:
        列 `date`, `Y1`〜`Y10`, `X1`〜`X50` の DataFrame（日付の昇順）。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    year_phase = 2 * np.pi * t / 365.25

    features: dict[str, np.ndarray] = {}
    # X1〜X10: 年周期・週周期・緩やかなトレンドを持つ連続値
    for i in range(1, 11):
        features[f"X{i}"] = (
            rng.normal(0, 1) * np.sin(year_phase + rng.uniform(0, 2 * np.pi))
            + 0.3 * np.sin(2 * np.pi * t / 7 + i)
            + rng.normal(0, 0.002) * t
            + rng.normal(0, 0.5, days)
        )
    # X11〜X20: 右に裾の長い正の値
    for i in range(11, 21):
        features[f"X{i}"] = rng.lognormal(
            mean=rng.uniform(0, 2), sigma=rng.uniform(0.6, 1.2), size=days
        )
    # X21〜X30: 欠損のある連続値
    for i in range(21, 31):
        values = rng.normal(rng.uniform(-5, 5), rng.uniform(0.5, 3), days)
        values[rng.random(days) < 0.02] = np.nan
        features[f"X{i}"] = values
    # X31〜X40: 件数
    for i in range(31, 41):
        features[f"X{i}"] = rng.poisson(rng.uniform(0.5, 10), days).astype(float)
    # X41〜X50: ノイズ
    for i in range(41, N_FEATURES + 1):
        features[f"X{i}"] = rng.normal(0, 1, days)

    targets = {
        col: _make_target(k, features, year_phase, rng) for k, col in enumerate(TARGET_COLS, 1)
    }
    frame = pd.DataFrame({"date": dates, **targets})
    return pd.concat([frame, pd.DataFrame(features)], axis=1)


def _standardize(values: np.ndarray) -> np.ndarray:
    """平均0・標準偏差1にそろえる（欠損は0＝平均とみなす）。"""
    centered = (values - np.nanmean(values)) / np.nanstd(values)
    return np.nan_to_num(centered)


def _make_target(
    k: int, features: dict[str, np.ndarray], year_phase: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """k 番目の目的変数を作る（前日の値への回帰 + 季節性 + 説明変数の効果 + ノイズ。正の値）。

    Args:
        k: 目的変数の番号（1〜10）。Xk・X(10+k)・X(20+k)・X(30+k) が効く。
        features: 説明変数。
        year_phase: 年周期の位相（ラジアン）。
        rng: 乱数生成器。

    Returns:
        目的変数の値。
    """
    days = len(year_phase)
    level = rng.uniform(10, 200)
    phi = rng.uniform(0.3, 0.9)  # 前日の値への回帰係数（自己相関の強さ）
    seasonal = level * rng.uniform(0.02, 0.25) * np.sin(year_phase + rng.uniform(0, 2 * np.pi))
    trend = level * 0.0003 * np.arange(days) if k % 3 == 0 else 0.0
    # 説明変数の効果（水準に比例した大きさ。符号・強さは目的変数ごとに乱数で決める）
    coefs = rng.uniform(0.02, 0.08, size=4) * rng.choice([-1, 1], size=4) * level
    effect = (
        coefs[0] * _standardize(features[f"X{k}"])
        + coefs[1] * _standardize(np.log(features[f"X{10 + k}"]))
        + coefs[2] * _standardize(features[f"X{20 + k}"])
        + coefs[3] * _standardize(features[f"X{30 + k}"])
    )
    noise = rng.normal(0, level * 0.03, days)
    base = level + seasonal + trend
    y = np.empty(days)
    value = level
    for i in range(days):
        value = phi * value + (1 - phi) * base[i] + effect[i] + noise[i]
        y[i] = max(value, level * 0.05)
    return y


def split_train_test(
    frame: pd.DataFrame, test_start: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """学習データ・テストデータ（目的変数なし）・テストの正解に分ける。

    Args:
        frame: `make_dummy_frame` の結果。
        test_start: テスト期間の開始日（この日以降がテスト）。

    Returns:
        (学習データ, テストデータ, テストの正解)。
    """
    is_test = frame["date"] >= pd.Timestamp(test_start)
    train = frame[~is_test].reset_index(drop=True)
    # テストデータには目的変数を入れない（予測時に分からない値のため）
    test = frame[is_test].drop(columns=TARGET_COLS).reset_index(drop=True)
    answer = frame.loc[is_test, ["date", *TARGET_COLS]].reset_index(drop=True)
    return train, test, answer


def main(argv: list[str] | None = None) -> None:
    """ダミーデータを作り、学習・テスト・正解の CSV に保存する。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2022-01-01", help="学習期間の開始日")
    parser.add_argument("--test-start", default="2025-01-01", help="テスト期間の開始日")
    parser.add_argument("--end", default="2025-12-31", help="テスト期間の終了日（この日を含む）")
    parser.add_argument("--seed", type=int, default=0, help="乱数のシード")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="出力先のディレクトリ")
    args = parser.parse_args(argv)

    days = (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days + 1
    # 学習・テストを1本の系列として作ってから分ける（テスト期間も同じ過程の続きになる）
    frame = make_dummy_frame(args.start, days, args.seed)
    parts = zip(
        (TRAIN_FILENAME, TEST_FILENAME, ANSWER_FILENAME),
        split_train_test(frame, args.test_start),
        strict=True,
    )
    for filename, part in parts:
        output = ensure_parent_dir(args.output_dir / filename)
        part.to_csv(output, index=False, date_format="%Y-%m-%d")
        dates = part["date"]
        print(
            f"保存: {output}（{len(part)}行 × {part.shape[1]}列、"
            f"{dates.min():%Y-%m-%d}〜{dates.max():%Y-%m-%d}）"
        )


if __name__ == "__main__":
    main()
