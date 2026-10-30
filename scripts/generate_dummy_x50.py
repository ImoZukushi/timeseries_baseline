"""特徴量エンジニアリングの動作確認用に、日次のダミーデータ（10種類の目的変数・X1〜X50）を作る。

1本の系列として作り、学習期間とテスト期間に分けて保存する（いずれも gitの管理外）。
データは long 型（1行 = 日付 × 目的変数の種類）で、目的変数の値を `Y`、種類（`Y1`〜`Y10`）を
カテゴリ変数 `Y_cat` に入れる。説明変数 X1〜X50 は日付ごとの値なので、同じ日付の行（10行）で
同じ値になる。
実データではないため `data/raw` には置かず、合成データ用の `data/synthetic/x50/` に置く。

X は `X1`〜`X50`。

| ファイル | 期間（既定） | 列 |
|---|---|---|
| `dummy_x50_train.csv` | 2022-01-01〜2024-12-31 | `date`, `Y_cat`, `irregular_flg`, `Y`, X |
| `dummy_x50_test.csv` | 2025-01-01〜2025-12-31 | `date`, `Y_cat`, `irregular_flg`, X（`Y` なし） |

`irregular_flg` は bool 型で、True の行は学習にも予測にも使わない（1行ごとに約3%）。
テストの正解ファイルは作らない。

行は日付の昇順、同じ日付の中は `Y1`, `Y2`, …, `Y10` の順。

説明変数は前処理の違いが分かるよう、性質の異なる5種類を10列ずつ用意する。

| 列 | 性質 |
|---|---|
| X1〜X10 | 季節性・トレンドのある連続値（正規分布に近い） |
| X11〜X20 | 右に裾の長い正の値（対数正規分布）。べき変換の対象になる |
| X21〜X30 | 連続値（散発的な欠損が他の列より多い。約2%を追加） |
| X31〜X40 | 件数（ポアソン分布の整数） |
| X41〜X50 | 目的変数と無関係なノイズ |

目的変数（`Y_cat` が Y1〜Y10 の `Y`）は、水準・一部の説明変数・ノイズから作る正の値で、
自己相関を持たない（前日の値への回帰・季節性・トレンドは入れない）。
モデルや前処理の違いが見えるよう、目的変数ごとに次を変える。

- 水準（10〜200）
- 効く説明変数: Yk には Xk・X(10+k)・X(20+k)・X(30+k) が効く
  （係数の符号・大きさも目的変数ごとに違う）。Xk は季節性・トレンドを持つが、それらが目的変数に
  移らないよう、Xk のランダムな成分だけが効く

自己相関が無いことは Ljung-Box 検定（帰無仮説: 自己相関なし）で確かめる。種類ごとに、欠損と
`irregular_flg` が True の行を除いた観測値の全期間・学習期間・テスト期間のそれぞれで、
ラグ 1, 7, 10, 14, 28 の p 値がすべて 0.10 を超える（有意水準 0.10 で帰無仮説が棄却されない）まで、
説明変数の係数とノイズを引き直す。

欠損: 目的変数・説明変数の両方に、散発的な欠損と連続した欠損を入れる（`add_missing`）。
値を作った後に「観測できなかった」ものとして欠損にするため、欠損の日も目的変数の値は
（観測されない）説明変数の値から作られている。欠損は学習期間・テスト期間の両方に入る。

| 対象 | 散発的な欠損 | 連続した欠損 |
|---|---|---|
| 目的変数（種類ごとに別々） | 約2% | 学習期間に2か所・テスト期間に1か所（各3〜35日） |
| 説明変数（列ごと。同じ日付の行で同じ） | 約1% | 学習に0〜2か所・テストに0〜1か所（各3〜30日） |

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
from statsmodels.stats.diagnostic import acorr_ljungbox

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from util.paths import data_dir, ensure_parent_dir  # noqa: E402

N_FEATURES = 50
N_TARGETS = 10
TARGET_COLS = [f"Y{k}" for k in range(1, N_TARGETS + 1)]  # 目的変数の種類（`Y_cat` の値）
CAT_COL = "Y_cat"  # long 型の表で、目的変数の種類を入れる列
VALUE_COL = "Y"  # long 型の表で、目的変数の値を入れる列
OUTPUT_DIR = data_dir() / "synthetic" / "x50"
TRAIN_FILENAME = "dummy_x50_train.csv"
TEST_FILENAME = "dummy_x50_test.csv"
IRREGULAR_COL = "irregular_flg"  # True の行は学習にも予測にも使わない
IRREGULAR_RATE = 0.03  # irregular_flg が True になる行の割合
LJUNG_BOX_ALPHA = 0.10  # 目的変数の自己相関の検定の有意水準
LJUNG_BOX_LAGS = (1, 7, 10, 14, 28)  # 検定するラグ（日数）
MAX_NOISE_DRAWS = 1000  # 検定を満たすまでノイズを引き直す最大回数


def make_dummy_frame(
    start: str = "2022-01-01",
    days: int = 1096,
    seed: int = 0,
    test_start: str = "2025-01-01",
) -> pd.DataFrame:
    """ダミーデータを作る（欠損は `add_missing` で入れる）。

    Args:
        start: 開始日。
        days: 日数（1行 = 1日）。
        seed: 乱数のシード。
        test_start: テスト期間の開始日（目的変数の自己相関を期間ごとに検定するために使う）。

    Returns:
        列 `date`, `Y1`〜`Y10`, `X1`〜`X50` の DataFrame（日付の昇順）。
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    t = np.arange(days)
    year_phase = 2 * np.pi * t / 365.25

    features: dict[str, np.ndarray] = {}
    # X1〜X10 のランダムな成分（目的変数にはこの成分だけを効かせる）
    random_parts: dict[int, np.ndarray] = {}
    # X1〜X10: 年周期・週周期・緩やかなトレンドを持つ連続値
    for i in range(1, 11):
        structured = (
            rng.normal(0, 1) * np.sin(year_phase + rng.uniform(0, 2 * np.pi))
            + 0.3 * np.sin(2 * np.pi * t / 7 + i)
            + rng.normal(0, 0.002) * t
        )
        random_parts[i] = rng.normal(0, 0.5, days)
        features[f"X{i}"] = structured + random_parts[i]
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

    # add_missing と同じ欠損の位置を使い、欠損と irregular_flg の行を除いて自己相関を検定する
    test_start_index = int((dates < pd.Timestamp(test_start)).sum())
    masks = _target_missing_masks(days, test_start_index, np.random.default_rng([seed, 1]))
    irregular = _irregular_masks(days, seed)
    targets = {
        col: _make_target(
            k, features, random_parts, masks[col] | irregular[col], test_start_index, rng
        )
        for k, col in enumerate(TARGET_COLS, 1)
    }
    frame = pd.DataFrame({"date": dates, **targets})
    return pd.concat([frame, pd.DataFrame(features)], axis=1)


def _standardize(values: np.ndarray) -> np.ndarray:
    """平均0・標準偏差1にそろえる（欠損は0＝平均とみなす）。"""
    centered = (values - np.nanmean(values)) / np.nanstd(values)
    return np.nan_to_num(centered)


def ljung_box_min_pvalue(values: np.ndarray) -> float:
    """欠損を除いた系列の Ljung-Box 検定の、`LJUNG_BOX_LAGS` の中で最小の p 値を返す。

    点数の半分以上のラグは検定しない。検定できるラグが無ければ 1.0 を返す。
    """
    observed = values[~np.isnan(values)]
    lags = [lag for lag in LJUNG_BOX_LAGS if lag < observed.size / 2]
    if not lags:
        return 1.0
    return float(acorr_ljungbox(observed, lags=lags)["lb_pvalue"].min())


def _has_no_autocorrelation(values: np.ndarray, test_start_index: int) -> bool:
    """全期間・学習期間・テスト期間のすべてで、検定が棄却されなければ True を返す。"""
    periods = (values, values[:test_start_index], values[test_start_index:])
    return all(ljung_box_min_pvalue(part) > LJUNG_BOX_ALPHA for part in periods)


def _make_target(
    k: int,
    features: dict[str, np.ndarray],
    random_parts: dict[int, np.ndarray],
    missing: np.ndarray,
    test_start_index: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """k 番目の目的変数を作る（水準 + 説明変数の効果 + ノイズ。自己相関の無い正の値）。

    Args:
        k: 目的変数の番号（1〜10）。Xk・X(10+k)・X(20+k)・X(30+k) が効く。
        features: 説明変数。
        random_parts: X1〜X10 のランダムな成分（番号 → 値）。
        missing: 検定から除く日（欠損になる日と irregular_flg が True の日）。
        test_start_index: テスト期間の最初の日の位置。
        rng: 乱数生成器。

    Returns:
        目的変数の値。

    Raises:
        RuntimeError: `MAX_NOISE_DRAWS` 回引き直しても検定を満たさない場合。
    """
    days = len(missing)
    level = rng.uniform(10, 200)
    # 説明変数の効果の元（どの成分も日ごとに独立なので、目的変数に自己相関は移らない）
    drivers = np.column_stack(
        [
            _standardize(random_parts[k]),
            _standardize(np.log(features[f"X{10 + k}"])),
            _standardize(features[f"X{20 + k}"]),
            _standardize(features[f"X{30 + k}"]),
        ]
    )
    # 自己相関が無い系列でも検定は有意水準の割合で棄却されるので、棄却されなくなるまで
    # 係数（水準に比例した大きさ。符号・強さは目的変数ごとに乱数で決める）とノイズを引き直す
    for _ in range(MAX_NOISE_DRAWS):
        coefs = rng.uniform(0.02, 0.08, size=4) * rng.choice([-1, 1], size=4) * level
        noise = rng.normal(0, level * 0.03, days)
        y = np.maximum(level + drivers @ coefs + noise, level * 0.05)
        if _has_no_autocorrelation(np.where(missing, np.nan, y), test_start_index):
            return y
    raise RuntimeError(f"Y{k}: {MAX_NOISE_DRAWS}回引き直しても自己相関の検定を満たしません")


def _missing_mask(
    days: int,
    test_start_index: int,
    rate: float,
    n_train_blocks: int,
    n_test_blocks: int,
    max_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """欠損にする日（True）を返す（散発的な欠損 + 連続した欠損）。

    Args:
        days: 日数。
        test_start_index: テスト期間の最初の日の位置（連続した欠損を学習・テストに分けて置く）。
        rate: 散発的な欠損の割合。
        n_train_blocks: 学習期間に置く連続した欠損の数。
        n_test_blocks: テスト期間に置く連続した欠損の数。
        max_length: 連続した欠損の最大日数（最小は3日）。
        rng: 乱数生成器。

    Returns:
        欠損にする日を True にした配列。
    """
    mask = rng.random(days) < rate
    for low, high, n_blocks in (
        (0, test_start_index, n_train_blocks),
        (test_start_index, days, n_test_blocks),
    ):
        for _ in range(n_blocks):
            length = int(rng.integers(3, max_length + 1))
            if high - low <= length:
                continue
            start = int(rng.integers(low, high - length))
            mask[start : start + length] = True
    return mask


def _target_missing_masks(
    days: int, test_start_index: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """目的変数の種類ごとの、欠損にする日を返す（種類ごとに別々の日）。"""
    return {col: _missing_mask(days, test_start_index, 0.02, 2, 1, 35, rng) for col in TARGET_COLS}


def _irregular_masks(days: int, seed: int) -> dict[str, np.ndarray]:
    """目的変数の種類ごとの、`irregular_flg` を True にする日を返す。

    値の生成・欠損とは別の乱数を使う。
    """
    rng = np.random.default_rng([seed, 2])
    return {col: rng.random(days) < IRREGULAR_RATE for col in TARGET_COLS}


def add_missing(frame: pd.DataFrame, test_start: str, seed: int = 0) -> pd.DataFrame:
    """wide 型のダミーデータの目的変数・説明変数に、散発的な欠損と連続した欠損を入れる。

    値を作った後に欠損にする（欠損の日も、目的変数は説明変数の真の値から作られている）。
    欠損の位置には値の生成とは別の乱数を使うため、欠損でない値は欠損を入れる前と同じ。

    Args:
        frame: `make_dummy_frame` の結果（wide 型）。
        test_start: テスト期間の開始日（連続した欠損を学習・テストの両方に置くために使う）。
        seed: 欠損の位置を決める乱数のシード。

    Returns:
        欠損を入れた DataFrame（入力は変更しない）。
    """
    rng = np.random.default_rng([seed, 1])
    days = len(frame)
    test_start_index = int((frame["date"] < pd.Timestamp(test_start)).sum())
    updates: dict[str, pd.Series] = {}
    # 目的変数: 種類ごとに別々の日を欠損にする
    for col, mask in _target_missing_masks(days, test_start_index, rng).items():
        updates[col] = frame[col].mask(mask)
    # 説明変数: 日付ごとの値なので、long 型にすると同じ日付の全種類の行で欠損になる
    feature_cols = [c for c in frame.columns if c not in ("date", *TARGET_COLS)]
    for col in feature_cols:
        n_train_blocks, n_test_blocks = int(rng.integers(0, 3)), int(rng.integers(0, 2))
        mask = _missing_mask(days, test_start_index, 0.01, n_train_blocks, n_test_blocks, 30, rng)
        updates[col] = frame[col].mask(mask)
    return frame.assign(**updates)


def to_long(frame: pd.DataFrame) -> pd.DataFrame:
    """wide 型（列 `date`, `Y1`〜`Y10`, `X1`〜`X50`）を long 型にする。

    long 型の列は `date`, `Y_cat`, `Y`, `X1`〜`X50`。

    行は日付の昇順、同じ日付の中は目的変数の種類の順（`Y1`, `Y2`, …）。
    """
    feature_cols = [c for c in frame.columns if c not in ("date", *TARGET_COLS)]
    long = frame.melt(
        id_vars=["date", *feature_cols],
        value_vars=TARGET_COLS,
        var_name=CAT_COL,
        value_name=VALUE_COL,
    )
    # 目的変数の種類の並び（Y1, Y2, …, Y10。文字列の並びでは Y10 が Y1 の次になるため番号で並べる）
    order = {name: i for i, name in enumerate(TARGET_COLS)}
    long = long.sort_values(
        ["date", CAT_COL], key=lambda s: s.map(order) if s.name == CAT_COL else s, kind="stable"
    )
    return long[["date", CAT_COL, VALUE_COL, *feature_cols]].reset_index(drop=True)


def split_train_test(
    frame: pd.DataFrame, test_start: str, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`irregular_flg` を付けた long 型の学習データ・テストデータ（目的変数の値なし）に分ける。

    Args:
        frame: `make_dummy_frame` の結果（wide 型）。
        test_start: テスト期間の開始日（この日以降がテスト）。
        seed: `make_dummy_frame` と同じ乱数のシード（同じ行を検定から除くため）。

    Returns:
        (学習データ, テストデータ)。いずれも long 型。
    """
    long = to_long(frame)
    flags = pd.Series(False, index=long.index, dtype=bool)
    # 行は日付の昇順なので、種類ごとの行の並びは日付の順
    for col, mask in _irregular_masks(frame["date"].nunique(), seed).items():
        flags[long[CAT_COL] == col] = mask
    long.insert(long.columns.get_loc(CAT_COL) + 1, IRREGULAR_COL, flags)
    is_test = long["date"] >= pd.Timestamp(test_start)
    train = long[~is_test].reset_index(drop=True)
    # テストデータには目的変数の値を入れない（予測時に分からない値のため）。種類（Y_cat）は残す
    test = long[is_test].drop(columns=VALUE_COL).reset_index(drop=True)
    return train, test


def main(argv: list[str] | None = None) -> None:
    """ダミーデータを作り、学習・テストの CSV に保存する。"""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2022-01-01", help="学習期間の開始日")
    parser.add_argument("--test-start", default="2025-01-01", help="テスト期間の開始日")
    parser.add_argument("--end", default="2025-12-31", help="テスト期間の終了日（この日を含む）")
    parser.add_argument("--seed", type=int, default=0, help="乱数のシード")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="出力先のディレクトリ")
    args = parser.parse_args(argv)

    days = (pd.Timestamp(args.end) - pd.Timestamp(args.start)).days + 1
    # 学習・テストを1本の系列として作ってから分ける（テスト期間も同じ過程の続きになる）
    frame = make_dummy_frame(args.start, days, args.seed, args.test_start)
    # 目的変数・説明変数に欠損を入れる（散発的な欠損と連続した欠損。学習・テストの両方）
    frame = add_missing(frame, args.test_start, args.seed)
    parts = zip(
        (TRAIN_FILENAME, TEST_FILENAME),
        split_train_test(frame, args.test_start, args.seed),
        strict=True,
    )
    irregular = _irregular_masks(days, args.seed)
    min_pvalue = min(
        ljung_box_min_pvalue(frame[col].mask(irregular[col]).to_numpy()) for col in TARGET_COLS
    )
    print(f"目的変数の Ljung-Box 検定の最小の p 値（全期間）: {min_pvalue:.3f}")
    for filename, part in parts:
        output = ensure_parent_dir(args.output_dir / filename)
        part.to_csv(output, index=False, date_format="%Y-%m-%d")
        dates = part["date"]
        missing = part.drop(columns=["date", CAT_COL, IRREGULAR_COL]).isna().mean()
        print(
            f"保存: {output}（{len(part)}行 × {part.shape[1]}列、"
            f"{dates.min():%Y-%m-%d}〜{dates.max():%Y-%m-%d}、"
            f"欠損率 {missing.min():.1%}〜{missing.max():.1%}）"
        )


if __name__ == "__main__":
    main()
