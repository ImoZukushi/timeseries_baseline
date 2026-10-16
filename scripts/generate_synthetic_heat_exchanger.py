"""熱交換器の汚れ予測用のダミーデータ（A部門）を生成し、`data/synthetic/` に保存するスクリプト。

原油の予熱トレインに並ぶ熱交換器10基（AA, AC, AE, AG, AI, O, R, T, W, Y。この順に高温側）を想定し、
2019-01-01〜2026-01-01 の日次データを、物理的な関係を持たせて乱数で作る。学習期間（2019〜2024年）と
予測期間（2025年）は1回の連続したシミュレーションから切り出す。

生成の考え方:
    - 原油26種（A〜Z）の性状は「重さ」の指標に連動させる（重いほど補正APIが低く、硫黄分・
      残留炭素分・アスファルテン・Ni/V・VR留分が多い）。体積割合6列の合計は 100 vol%。
    - 日々の原油構成は、7〜21日ごとに切り替わる配合（3〜5種）で、配合比率はゆっくり変動する。
    - 汚れ係数（m²·K/W）は日ごとに「付着 − 除去」で増える
      （Ebert-Panchal / Kern-Seaton 型の簡易モデル）。
      付着は、境膜温度（Hot側）が高いほど（アレニウス型）・原油の汚れやすさ（配合比率で加重平均した
      アスファルテン・残留炭素分・VR留分・Ni+V）が大きいほど速く、汚れ防止剤（注入量で飽和する抑制。
      種類 A は C より効く）と苛性ソーダで抑えられ、デソルター温度が低いと少し速くなる。
      除去は、せん断応力（Tube側）× 汚れ係数に比例する。
    - 汚れると伝熱が落ちて境膜温度（Cold側）が下がり、流路が狭まってせん断応力（Tube側）が上がる。
    - 汚れ係数が機器ごとの閾値を超えるか、前回の洗浄から一定の日数（300〜450日）がたつと、
      3〜7日止めて洗浄し、小さな値に戻る（のこぎり形）。ほかに突発停止（1〜3日。洗浄なし）がある。停止中は運転条件・汚れ係数が欠測になる。
      洗浄・停止の記録は予測期間（2025年）の分だけ出力する。
    - 汚れ係数の観測値は、真値に測定ノイズを加えたもの。ランダムな欠測（約1%）と、
      スパイク状の異常値（約1%）を混ぜる。

出力（`data/synthetic/`。日付は YYYY-MM-DD、縦持ち）:
    - `薬剤_A部門.csv`: 日付, 汚れ防止剤, 苛性ソーダ, 汚れ防止剤種類（A / C）
    - `目的変数_学習期間_A部門.csv`: 日付, 熱交換器, 汚れ係数, 汚れ係数の翌日増加量, 異常値flg
    - `目的変数_予測期間_A部門.csv`: ID, 日付, 熱交換器, 汚れ係数（null）,
      汚れ係数の翌日増加量（null）,
      異常値flg
    - `目的変数_予測期間_正解_A部門.csv`: 予測期間の正解（列は予測期間のファイルと同じ）
    - `熱交換器_運転条件_A部門.csv`: 日付, 熱交換器, 境膜温度(Cold-side), 境膜温度(Hot-side),
      せん断応力(Tube-side), せん断応力(Shell-side)
    - `洗浄・装置停止期間_A部門.csv`: 日付, 熱交換器, 洗浄日flg, 装置停止flg（2025年のみ）
    - `原油構成_A部門.csv`: 日付, 原油コード, 原油配合比率（同じ日付の合計が 1.0）
    - `原油性状_A部門.csv`: 原油コードごとの性状（27列）
    - `その他運転条件_A部門.csv`: 日付, プレカラム下流圧力_タグ3/4/5, プレカラム温度, デソルター温度

定義:
    - 汚れ係数の翌日増加量: 当日の行に「翌日の観測値 − 当日の観測値」を入れる
      （どちらかが欠測なら欠測）。
    - 異常値flg: 汚れ係数か翌日増加量が欠測の行、または異常値を入れた行で 1。それ以外は 0。
    - 洗浄日flg: 洗浄のための停止の日で 1。装置停止flg: 停止中（洗浄・突発停止）の日で 1。

Usage:
    uv run python scripts/generate_synthetic_heat_exchanger.py
    uv run python scripts/generate_synthetic_heat_exchanger.py --seed 1 --output-dir data/synthetic
"""

from __future__ import annotations

import argparse
import datetime as dt
import string
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

# scripts/ から直接実行しても src/ 配下のパッケージをimportできるようにする
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from util.paths import data_dir  # noqa: E402

# --- 期間・名前 ------------------------------------------------------------------------

START = dt.date(2019, 1, 1)
TRAIN_END = dt.date(2024, 12, 31)  # 学習期間の終わり
TEST_END = dt.date(2025, 12, 31)  # 予測期間（TRAIN_END の翌日から）の終わり
# 運転条件などの終わり（予測期間の最終日の翌日増加量に必要な翌日まで）
CONDITION_END = TEST_END + dt.timedelta(days=1)

EXCHANGERS = ("AA", "AC", "AE", "AG", "AI", "O", "R", "T", "W", "Y")
CRUDE_CODES = tuple(string.ascii_uppercase)  # A〜Z
ANTIFOULANT_TYPES = ("A", "C")

DEPARTMENT = "A部門"
FILE_NAMES = {
    "chemical": f"薬剤_{DEPARTMENT}.csv",
    "target_train": f"目的変数_学習期間_{DEPARTMENT}.csv",
    "target_test": f"目的変数_予測期間_{DEPARTMENT}.csv",
    "target_test_answer": f"目的変数_予測期間_正解_{DEPARTMENT}.csv",
    "exchanger_conditions": f"熱交換器_運転条件_{DEPARTMENT}.csv",
    "cleaning_stops": f"洗浄・装置停止期間_{DEPARTMENT}.csv",
    "crude_composition": f"原油構成_{DEPARTMENT}.csv",
    "crude_properties": f"原油性状_{DEPARTMENT}.csv",
    "other_conditions": f"その他運転条件_{DEPARTMENT}.csv",
}

# --- 列名 ------------------------------------------------------------------------------

DATE = "日付"
EXCHANGER = "熱交換器"
FOULING = "汚れ係数"
FOULING_INCREASE = "汚れ係数の翌日増加量"
ANOMALY_FLAG = "異常値flg"
TARGET_COLUMNS = (DATE, EXCHANGER, FOULING, FOULING_INCREASE, ANOMALY_FLAG)
CONDITION_COLUMNS = (
    "境膜温度(Cold-side)",
    "境膜温度(Hot-side)",
    "せん断応力(Tube-side)",
    "せん断応力(Shell-side)",
)
# 運転条件の値の範囲
CONDITION_RANGES = ((0.0, 250.0), (0.0, 300.0), (0.0, 5.0), (0.0, 5.0))
OTHER_CONDITION_RANGES = {
    "プレカラム下流圧力_タグ3": (250.0, 350.0),
    "プレカラム下流圧力_タグ4": (1500.0, 1800.0),
    "プレカラム下流圧力_タグ5": (1400.0, 1700.0),
    "プレカラム温度": (100.0, 130.0),
    "デソルター温度": (90.0, 100.0),
}
ANTIFOULANT_RANGE = (-0.02, 40.0)
CAUSTIC_RANGE = (-0.1, 40.0)
FOULING_RANGE = (0.0, 0.01)

# 原油性状: 列名 → (下限, 上限, 重さとの関係。+1 なら重いほど大きい、-1 なら小さい、0 なら関係なし)
CRUDE_PROPERTY_SPECS: dict[str, tuple[float, float, int]] = {
    "補正API": (20.0, 45.0, -1),
    "硫黄分[mass%]": (0.0, 4.5, 1),
    "窒素分[mass%]": (0.0, 0.2, 1),
    "残留炭素分[mass%]": (0.0, 4.5, 1),
    "動粘度[mm2/s]": (0.0, 100.0, 1),
    "TAN[mgKOH/g]": (0.0, 1.5, 0),
    "アスファルテン[mass%]": (0.0, 10.0, 1),
    "Ni[ppm]": (0.0, 10.0, 1),
    "V[ppm]": (0.0, 10.0, 1),
}
# 体積割合（留分）: 合計が 100 vol% になるよう調整する
VOLUME_SPECS: dict[str, tuple[float, float, int]] = {
    "体積割合_LPG+LSR[vol%]": (0.0, 30.0, -1),
    "体積割合_HSR[vol%]": (5.0, 30.0, -1),
    "体積割合_KERO[vol%]": (20.0, 30.0, -1),
    "体積割合_LGO[vol%]": (5.0, 30.0, 0),
    "体積割合_VGO[vol%]": (0.0, 30.0, 1),
    "体積割合_VR[vol%]": (0.0, 15.0, 1),
}
# 密度（値の大きさは指定どおり 0.7〜1.1。単位表記は指定の kg/m3 のまま）
DENSITY_SPECS: dict[str, tuple[float, float, int]] = {
    "密度_LSR[kg/m3]": (0.7, 0.8, 1),
    "密度_HSR[kg/m3]": (0.7, 0.8, 1),
    "密度_KERO[kg/m3]": (0.7, 0.8, 1),
    "密度_LGO[kg/m3]": (0.7, 0.8, 1),
    "密度_VGO[kg/m3]": (0.7, 0.8, 1),
    "密度_VR[kg/m3]": (0.7, 1.1, 1),
}
FRACTION_SPECS: dict[str, tuple[float, float, int]] = {
    "VGO_窒素分[mass%]": (0.01, 0.2, 1),
    "VGO_硫黄分[mass%]": (0.01, 0.2, 1),
    "VGO_残留炭素分[mass%]": (0.01, 1.0, 1),
    "VR_窒素分[mass%]": (0.01, 0.2, 1),
    "VR_硫黄分[mass%]": (0.01, 0.2, 1),
    "VR_残留炭素分[mass%]": (0.01, 10.0, 1),
}
CRUDE_PROPERTY_COLUMNS = (
    *CRUDE_PROPERTY_SPECS,
    *VOLUME_SPECS,
    *DENSITY_SPECS,
    *FRACTION_SPECS,
)

# --- モデルのパラメータ -----------------------------------------------------------------------

# 汚れ防止剤の最大の抑制率（種類 A は C より効く）と、効きが半分になる注入量
ANTIFOULANT_MAX_INHIBITION = {"A": 0.6, "C": 0.4}
ANTIFOULANT_HALF_DOSE = 10.0
# 付着の活性化エネルギー（J/mol）と基準温度（℃）
ACTIVATION_ENERGY = 40_000.0
REFERENCE_TEMPERATURE = 230.0
GAS_CONSTANT = 8.314
# 除去項の係数（せん断応力 × 汚れ係数 に掛ける）
REMOVAL_COEFFICIENT = 7.0e-4
# 汚れ係数の測定ノイズ（標準偏差）と、ランダムな欠測・異常値の割合
MEASUREMENT_NOISE = 1.2e-4
MISSING_RATE = 0.01
ANOMALY_RATE = 0.01
CONDITION_MISSING_RATE = 0.01


@dataclass(frozen=True)
class ExchangerSpec:
    """1基の熱交換器の特性。"""

    name: str
    hot_film: float  # Hot側の境膜温度の基準値（℃）
    cold_film: float  # Cold側の境膜温度の基準値（℃）
    tube_shear: float  # Tube側のせん断応力の基準値（Pa）
    shell_shear: float  # Shell側のせん断応力の基準値（Pa）
    deposition: float  # 付着の速さの係数（1日あたり）
    threshold: float  # 洗浄する汚れ係数
    residual: float  # 洗浄後に残る汚れ係数
    max_interval: int  # 閾値に届かなくても計画的に洗浄するまでの日数


@dataclass
class SyntheticData:
    """生成したダミーデータ。

    Attributes:
        frames: ファイルの種類（`FILE_NAMES` のキー）→ 出力する DataFrame。
        truth: 確認用の内部の値（日付・熱交換器ごとの汚れ係数の真値・付着速度・停止の種類）。
            ファイルには出力しない。
    """

    frames: dict[str, pl.DataFrame]
    truth: pl.DataFrame = field(default_factory=pl.DataFrame)


# --- 原油 ------------------------------------------------------------------------------


def _scaled(
    rng: np.random.Generator, heaviness: np.ndarray, spec: tuple[float, float, int], noise: float
) -> np.ndarray:
    """重さの指標（0〜1）を、性状の範囲 [下限, 上限] に対応づける（ばらつきを加える）。"""
    low, high, direction = spec
    if direction == 0:
        t = rng.uniform(0.05, 0.95, heaviness.size)
    else:
        base = heaviness if direction > 0 else 1.0 - heaviness
        t = base + rng.normal(0.0, noise, heaviness.size)
    return low + np.clip(t, 0.02, 0.98) * (high - low)


def _fit_to_total(
    values: np.ndarray, lows: np.ndarray, highs: np.ndarray, total: float
) -> np.ndarray:
    """各値を範囲 [lows, highs] に収めたまま、合計が total になるよう調整する。

    足りない（余る）分を、各値が上限（下限）まで動ける幅に比例して配る。
    合計が取りうる範囲（lows の合計〜highs の合計）に total が入っていれば、1回で合う。
    """
    diff = total - values.sum()
    room = (highs - values) if diff > 0 else (values - lows)
    return values + diff * room / room.sum()


def crude_properties(rng: np.random.Generator) -> pl.DataFrame:
    """原油26種（A〜Z）の性状を作る。

    重さの指標（0〜1）を原油ごとに決め、各性状を指定の範囲内で連動させる。
    体積割合6列は、範囲内に収めたまま合計を 100 vol% にする。

    Args:
        rng: 乱数生成器。

    Returns:
        1行 = 1原油。列は `原油コード` と `CRUDE_PROPERTY_COLUMNS`。
    """
    n = len(CRUDE_CODES)
    heaviness = rng.uniform(0.0, 1.0, n)
    columns: dict[str, np.ndarray] = {}
    for name, spec in {**CRUDE_PROPERTY_SPECS, **DENSITY_SPECS, **FRACTION_SPECS}.items():
        columns[name] = _scaled(rng, heaviness, spec, noise=0.12)
    # 動粘度は重質原油で急に大きくなる
    low, high, _ = CRUDE_PROPERTY_SPECS["動粘度[mm2/s]"]
    columns["動粘度[mm2/s]"] = low + (high - low) * np.clip(
        heaviness**2.5 + rng.normal(0.0, 0.03, n), 0.01, 0.99
    )

    lows = np.array([spec[0] for spec in VOLUME_SPECS.values()])
    highs = np.array([spec[1] for spec in VOLUME_SPECS.values()])
    raw = np.column_stack(
        [_scaled(rng, heaviness, spec, noise=0.15) for spec in VOLUME_SPECS.values()]
    )
    volumes = np.vstack([_fit_to_total(row, lows, highs, 100.0) for row in raw])
    volumes = np.round(volumes, 2)
    # 丸めの誤差は、範囲に余裕のある LGO で吸収する（合計をちょうど 100 にする）
    lgo = list(VOLUME_SPECS).index("体積割合_LGO[vol%]")
    volumes[:, lgo] = np.round(100.0 - (volumes.sum(axis=1) - volumes[:, lgo]), 2)
    for j, name in enumerate(VOLUME_SPECS):
        columns[name] = volumes[:, j]

    frame = pl.DataFrame(
        {"原油コード": list(CRUDE_CODES), **{c: columns[c] for c in CRUDE_PROPERTY_COLUMNS}}
    )
    # 表示桁をそろえる（体積割合は合計を保つため丸め済み）
    return frame.with_columns(
        [pl.col(c).round(4) for c in CRUDE_PROPERTY_COLUMNS if c not in VOLUME_SPECS]
    )


def crude_fouling_index(properties: pl.DataFrame) -> np.ndarray:
    """原油ごとの汚れやすさ（0〜1 程度）。アスファルテン・残留炭素分・VR留分・Ni+V の加重和。"""
    p = properties
    return (
        0.35 * p["アスファルテン[mass%]"].to_numpy() / 10.0
        + 0.25 * p["残留炭素分[mass%]"].to_numpy() / 4.5
        + 0.25 * p["体積割合_VR[vol%]"].to_numpy() / 15.0
        + 0.15 * (p["Ni[ppm]"].to_numpy() + p["V[ppm]"].to_numpy()) / 20.0
    )


def crude_slate(rng: np.random.Generator, n_days: int) -> np.ndarray:
    """日ごとの原油配合比率（n_days × 26。各行の合計は 1.0、使わない原油は 0）。

    7〜21日ごとに配合（3〜5種）を切り替え、その間の比率はゆっくり変動させる。
    比率は小数4桁に丸め、最も多い原油で合計をちょうど 1.0 にそろえる。
    """
    ratios = np.zeros((n_days, len(CRUDE_CODES)))
    day = 0
    while day < n_days:
        length = int(rng.integers(7, 22))
        k = int(rng.integers(3, 6))
        codes = rng.choice(len(CRUDE_CODES), size=k, replace=False)
        base = rng.dirichlet(np.full(k, 2.0))
        drift = np.zeros(k)
        for d in range(day, min(day + length, n_days)):
            drift += rng.normal(0.0, 0.04, k)
            weights = base * np.exp(drift)
            weights = np.maximum(np.round(weights / weights.sum(), 4), 0.0001)
            # 丸めた後の合計を 1.0 にそろえる（最も多い原油で調整する）
            weights[np.argmax(weights)] += 1.0 - weights.sum()
            ratios[d, codes] = np.round(weights, 4)
        day += length
    return ratios


def _composition_frame(dates: list[dt.date], ratios: np.ndarray) -> pl.DataFrame:
    """配合比率の行列を、縦持ち（日付, 原油コード, 原油配合比率。使う原油だけ）にする。"""
    rows_d, rows_c = np.nonzero(ratios)
    return pl.DataFrame(
        {
            DATE: [dates[i] for i in rows_d],
            "原油コード": [CRUDE_CODES[j] for j in rows_c],
            "原油配合比率": ratios[rows_d, rows_c],
        }
    ).sort(DATE, "原油コード")


# --- 部門単位の時系列 --------------------------------------------------------------------------


def _mean_reverting(
    rng: np.random.Generator, n: int, mean: float, sd: float, phi: float = 0.97
) -> np.ndarray:
    """平均 mean・定常の標準偏差 sd の AR(1) 系列（平均に戻る、ゆっくりした変動）。"""
    shocks = rng.normal(0.0, sd * np.sqrt(1.0 - phi**2), n)
    values = np.empty(n)
    values[0] = mean + rng.normal(0.0, sd)
    for i in range(1, n):
        values[i] = mean + phi * (values[i - 1] - mean) + shocks[i]
    return values


def other_conditions(rng: np.random.Generator, n_days: int) -> dict[str, np.ndarray]:
    """部門単位のその他運転条件（範囲の中央付近を平均に、範囲内で変動させる）。"""
    out = {}
    for name, (low, high) in OTHER_CONDITION_RANGES.items():
        mean, sd = (low + high) / 2.0, (high - low) / 8.0
        out[name] = np.clip(_mean_reverting(rng, n_days, mean, sd), low, high)
    return out


def chemicals(
    rng: np.random.Generator, n_days: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """汚れ防止剤・苛性ソーダの注入量（真値と計器の値）と、汚れ防止剤の種類。

    汚れ防止剤の種類は 60〜150日ごとに A ⇄ C を切り替え、注入量は種類の期間ごとの基準値の
    周りでゆっくり変動する。注入を止める期間（5〜20日）もある。計器の値は真値にノイズを
    加えたもので、注入量が 0 の日は 0 付近の小さな値（負になることもある）になる。

    Returns:
        (汚れ防止剤の真値, 汚れ防止剤の計器の値, 苛性ソーダの真値, 苛性ソーダの計器の値, 種類)。
    """
    kinds = np.empty(n_days, dtype=object)
    dose = np.empty(n_days)
    day = 0
    kind_index = int(rng.integers(0, 2))
    while day < n_days:
        length = int(rng.integers(60, 151))
        end = min(day + length, n_days)
        kinds[day:end] = ANTIFOULANT_TYPES[kind_index]
        dose[day:end] = _mean_reverting(rng, end - day, rng.uniform(8.0, 30.0), 3.0)
        if rng.random() < 0.3 and end - day > 30:
            # 注入を止める期間
            stop = int(rng.integers(day, end - 20))
            dose[stop : stop + int(rng.integers(5, 21))] = 0.0
        kind_index = 1 - kind_index
        day = end
    dose = np.clip(dose, 0.0, 39.5)
    caustic = np.clip(_mean_reverting(rng, n_days, 15.0, 5.0), 0.0, 39.5)
    for _ in range(int(rng.integers(1, 4))):
        stop = int(rng.integers(0, max(n_days - 15, 1)))
        caustic[stop : stop + int(rng.integers(3, 15))] = 0.0

    def measured(true: np.ndarray, low: float) -> np.ndarray:
        # 注入していない日は 0 付近の小さなノイズ（下限まで負になりうる）、注入中は相対ノイズ
        noise = np.where(
            true == 0.0,
            rng.normal(0.0, abs(low) / 2.0, true.size),
            rng.normal(0.0, 0.3, true.size),
        )
        return np.round(np.clip(true + noise, low, 40.0), 3)

    return (
        dose,
        measured(dose, ANTIFOULANT_RANGE[0]),
        caustic,
        measured(caustic, CAUSTIC_RANGE[0]),
        kinds,
    )


# --- 熱交換器 -----------------------------------------------------------------------------


def make_exchangers(rng: np.random.Generator) -> list[ExchangerSpec]:
    """熱交換器10基の特性（下流ほど高温で、付着が速い）。"""
    specs = []
    for i, name in enumerate(EXCHANGERS):
        specs.append(
            ExchangerSpec(
                name=name,
                hot_film=float(185.0 + 9.0 * i + rng.uniform(-5.0, 5.0)),
                cold_film=float(125.0 + 9.0 * i + rng.uniform(-5.0, 5.0)),
                tube_shear=float(rng.uniform(1.5, 3.5)),
                shell_shear=float(rng.uniform(1.0, 3.0)),
                deposition=float(rng.uniform(6.0e-5, 9.0e-5)),
                threshold=float(rng.uniform(0.006, 0.009)),
                residual=float(rng.uniform(0.0005, 0.0015)),
                max_interval=int(rng.integers(300, 451)),
            )
        )
    return specs


def _arrhenius(temperature: np.ndarray | float) -> np.ndarray | float:
    """基準温度に対する付着速度の倍率（アレニウス型）。"""
    t = np.asarray(temperature) + 273.15
    t_ref = REFERENCE_TEMPERATURE + 273.15
    return np.exp(-ACTIVATION_ENERGY / GAS_CONSTANT * (1.0 / t - 1.0 / t_ref))


@dataclass
class _ExchangerRun:
    """1基分のシミュレーション結果（日ごとの配列）。"""

    fouling: np.ndarray  # 汚れ係数の真値
    deposition: np.ndarray  # その日の付着速度（除去前）
    conditions: np.ndarray  # 運転条件（n_days × 4。停止日は NaN）
    stop_kind: np.ndarray  # 0: 運転、1: 洗浄のための停止、2: 突発停止


def simulate_exchanger(
    rng: np.random.Generator,
    spec: ExchangerSpec,
    throughput: np.ndarray,
    foulant: np.ndarray,
    inhibition: np.ndarray,
    external: np.ndarray,
) -> _ExchangerRun:
    """1基の熱交換器の汚れ係数・運転条件・停止を日ごとに計算する。

    Args:
        rng: 乱数生成器。
        spec: 熱交換器の特性。
        throughput: 処理量の倍率（日ごと、1.0 前後）。
        foulant: 原油の汚れやすさ（配合比率で加重平均。日ごと）。
        inhibition: 薬剤（汚れ防止剤・苛性ソーダ）による付着の抑制率（日ごと、0〜1）。
        external: その他運転条件による付着の倍率（日ごと、1.0 前後）。

    Returns:
        日ごとの配列。
    """
    n = throughput.size
    fouling = np.empty(n)
    deposition = np.zeros(n)
    conditions = np.full((n, 4), np.nan)
    stop_kind = np.zeros(n, dtype=np.int64)
    r = float(rng.uniform(spec.residual, 0.8 * spec.threshold))  # 期間の初日の汚れ係数
    cleaning_left = trip_left = 0
    days_since_cleaning = int(rng.integers(0, spec.max_interval))
    for d in range(n):
        if cleaning_left > 0:
            stop_kind[d] = 1
            cleaning_left -= 1
            if cleaning_left == 0:
                r = spec.residual * float(rng.uniform(0.8, 1.2))  # 洗浄で小さな値に戻る
                days_since_cleaning = 0
            fouling[d] = r
            continue
        if trip_left > 0:
            stop_kind[d] = 2
            trip_left -= 1
            fouling[d] = r
            continue
        q = throughput[d]
        hot = spec.hot_film * (0.97 + 0.03 * q) + rng.normal(0.0, 2.0)
        # 汚れると伝熱が落ちて Cold 側の境膜温度が下がり、流路が狭まって Tube 側のせん断応力が上がる
        cold = spec.cold_film * (0.98 + 0.02 * q) - 2000.0 * r + rng.normal(0.0, 2.0)
        tube = spec.tube_shear * q**2 * (1.0 + 40.0 * r) + rng.normal(0.0, 0.05)
        shell = spec.shell_shear * q**2 + rng.normal(0.0, 0.05)
        conditions[d] = [
            np.clip(cold, *CONDITION_RANGES[0]),
            np.clip(hot, *CONDITION_RANGES[1]),
            np.clip(tube, *CONDITION_RANGES[2]),
            np.clip(shell, *CONDITION_RANGES[3]),
        ]
        dep = (
            spec.deposition
            * float(_arrhenius(hot))
            * (0.3 + foulant[d])
            * (1.0 - inhibition[d])
            * external[d]
        )
        deposition[d] = dep
        r = max(r + dep * float(rng.lognormal(0.0, 0.2)) - REMOVAL_COEFFICIENT * tube * r, 0.0)
        fouling[d] = r
        days_since_cleaning += 1
        # 閾値を超えるか、前回の洗浄から max_interval 日たったら翌日から洗浄する。
        # まれに突発停止もある（初日と2日目は止めない）
        if d >= 1:
            if r > spec.threshold or days_since_cleaning >= spec.max_interval:
                cleaning_left = int(rng.integers(3, 8))
            elif rng.random() < 1.0 / 365.0:
                trip_left = int(rng.integers(1, 4))
    return _ExchangerRun(fouling, deposition, conditions, stop_kind)


# --- 目的変数 ---------------------------------------------------------------------------


def observe_fouling(
    rng: np.random.Generator, fouling: np.ndarray, stopped: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """汚れ係数の観測値と、異常値を入れたかの印を作る。

    観測値 = 真値 + 測定ノイズ（0〜0.01 に収め、小数6桁に丸める）。停止日とランダムな欠測は NaN。
    欠測でない日の約1%に、スパイク状の異常値（真値の 1.6〜3倍、または 0.2〜0.5倍）を入れる。
    期間の初日と2日目は欠測・異常値にしない（初日の翌日増加量を必ず計算できるようにする）。
    """
    n = fouling.size
    observed = fouling + rng.normal(0.0, MEASUREMENT_NOISE, n)
    observed = np.clip(observed, *FOULING_RANGE)
    missing = stopped | (rng.random(n) < MISSING_RATE)
    missing[:2] = False
    observed[missing] = np.nan
    anomaly = (~missing) & (rng.random(n) < ANOMALY_RATE)
    anomaly[:2] = False
    scale = np.where(rng.random(n) < 0.5, rng.uniform(1.6, 3.0, n), rng.uniform(0.2, 0.5, n))
    observed = np.where(anomaly, np.clip(observed * scale, *FOULING_RANGE), observed)
    return np.round(observed, 6), anomaly


def target_frame(
    dates: list[dt.date], name: str, observed: np.ndarray, anomaly: np.ndarray
) -> pl.DataFrame:
    """1基分の目的変数（汚れ係数・翌日増加量・異常値flg）。最終日は翌日が無いので除く。"""
    increase = np.round(observed[1:] - observed[:-1], 6)  # 欠測を含めば NaN になる
    current = observed[:-1]
    flag = np.isnan(current) | np.isnan(increase) | anomaly[:-1]
    return pl.DataFrame(
        {
            DATE: dates[:-1],
            EXCHANGER: name,
            FOULING: current,
            FOULING_INCREASE: increase,
            ANOMALY_FLAG: flag.astype(np.int64),
        }
    ).with_columns(pl.col(FOULING, FOULING_INCREASE).fill_nan(None))


# --- 全体 ------------------------------------------------------------------------------


def generate(
    seed: int = 42,
    start: dt.date = START,
    train_end: dt.date = TRAIN_END,
    test_end: dt.date = TEST_END,
) -> SyntheticData:
    """ダミーデータ一式を生成する。

    start〜（test_end の翌日）を1回の連続したシミュレーションで計算し、学習期間
    （start〜train_end）と
    予測期間（train_end の翌日〜test_end）に切り出す。

    Args:
        seed: 乱数シード。
        start: 期間の初日。
        train_end: 学習期間の最終日。
        test_end: 予測期間の最終日。

    Returns:
        出力する DataFrame 一式と、確認用の内部の値。

    Raises:
        ValueError: 期間の指定が start < train_end < test_end になっていない場合。
    """
    if not start < train_end < test_end:
        raise ValueError("期間は start < train_end < test_end で指定してください")
    rng = np.random.default_rng(seed)
    condition_end = test_end + dt.timedelta(days=1)
    dates = pl.date_range(start, condition_end, interval="1d", eager=True).to_list()
    n_days = len(dates)
    test_start = train_end + dt.timedelta(days=1)

    properties = crude_properties(rng)
    ratios = crude_slate(rng, n_days)
    foulant = ratios @ crude_fouling_index(properties)
    others = other_conditions(rng, n_days)
    dose, dose_measured, caustic, caustic_measured, kinds = chemicals(rng, n_days)
    max_inhibition = np.array([ANTIFOULANT_MAX_INHIBITION[k] for k in kinds])
    inhibition = 1.0 - (1.0 - max_inhibition * dose / (dose + ANTIFOULANT_HALF_DOSE)) * (
        1.0 - 0.1 * caustic / (caustic + 10.0)
    )
    # デソルター温度が低いと塩分が残り、付着が少し速くなる
    desalter = others["デソルター温度"]
    external = 1.0 + 0.05 * (95.0 - desalter) / 5.0
    throughput = np.clip(_mean_reverting(rng, n_days, 1.0, 0.05), 0.8, 1.2)

    target_parts, condition_parts, stop_parts, truth_parts = [], [], [], []
    for spec in make_exchangers(rng):
        run = simulate_exchanger(rng, spec, throughput, foulant, inhibition, external)
        stopped = run.stop_kind > 0
        observed, anomaly = observe_fouling(rng, run.fouling, stopped)
        target_parts.append(target_frame(dates, spec.name, observed, anomaly))
        conditions = run.conditions.copy()
        # センサーの欠測（列ごとに独立）
        conditions[rng.random(conditions.shape) < CONDITION_MISSING_RATE] = np.nan
        decimals = (2, 2, 3, 3)
        condition_parts.append(
            pl.DataFrame(
                {
                    DATE: dates,
                    EXCHANGER: spec.name,
                    **{
                        c: np.round(conditions[:, j], decimals[j])
                        for j, c in enumerate(CONDITION_COLUMNS)
                    },
                }
            ).with_columns(pl.col(*CONDITION_COLUMNS).fill_nan(None))
        )
        stop_parts.append(
            pl.DataFrame(
                {
                    DATE: dates,
                    EXCHANGER: spec.name,
                    "洗浄日flg": (run.stop_kind == 1).astype(np.int64),
                    "装置停止flg": stopped.astype(np.int64),
                }
            )
        )
        truth_parts.append(
            pl.DataFrame(
                {
                    DATE: dates,
                    EXCHANGER: spec.name,
                    "汚れ係数_真値": run.fouling,
                    "付着速度": run.deposition,
                    "停止の種類": run.stop_kind,
                    "原油の汚れやすさ": foulant,
                    "汚れ防止剤_真値": dose,
                }
            )
        )

    targets = pl.concat(target_parts).sort(DATE, EXCHANGER)
    train = targets.filter(pl.col(DATE) <= train_end)
    answer = (
        targets.filter(pl.col(DATE) >= test_start)
        .with_row_index("ID")
        .with_columns(pl.col("ID").cast(pl.Int64))
        .select("ID", *TARGET_COLUMNS)
    )
    test = answer.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(FOULING),
        pl.lit(None, dtype=pl.Float64).alias(FOULING_INCREASE),
    )
    stops = (
        pl.concat(stop_parts)
        .filter(pl.col(DATE).is_between(test_start, test_end))
        .sort(DATE, EXCHANGER)
    )
    frames = {
        "chemical": pl.DataFrame(
            {
                DATE: dates,
                "汚れ防止剤": dose_measured,
                "苛性ソーダ": caustic_measured,
                "汚れ防止剤種類": kinds.tolist(),
            }
        ),
        "target_train": train,
        "target_test": test,
        "target_test_answer": answer,
        "exchanger_conditions": pl.concat(condition_parts).sort(DATE, EXCHANGER),
        "cleaning_stops": stops,
        "crude_composition": _composition_frame(dates, ratios),
        "crude_properties": properties,
        "other_conditions": pl.DataFrame(
            {DATE: dates, **{k: np.round(v, 2) for k, v in others.items()}}
        ),
    }
    return SyntheticData(frames=frames, truth=pl.concat(truth_parts).sort(DATE, EXCHANGER))


def _check_output_dir(path: Path) -> None:
    """出力先が data/raw ・ data/external の下でないことを確かめる（元データを上書きしない）。"""
    resolved = path.resolve()
    for protected in (data_dir() / "raw", data_dir() / "external"):
        if resolved.is_relative_to(protected.resolve()):
            raise ValueError(f"data/raw・data/external には書き込めません: {path}")


def write_files(data: SyntheticData, out_dir: Path) -> list[Path]:
    """生成したデータを CSV（UTF-8）で保存する。

    Args:
        data: `generate` の出力。
        out_dir: 保存先ディレクトリ。

    Returns:
        保存したファイルのパス。

    Raises:
        ValueError: 保存先が data/raw・data/external の下の場合。
    """
    _check_output_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, filename in FILE_NAMES.items():
        path = out_dir / filename
        data.frames[key].write_csv(path)
        paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> None:
    """ダミーデータを生成して保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（既定: 42）")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="保存先ディレクトリ（既定: data/synthetic）"
    )
    args = parser.parse_args(argv)
    out_dir = args.output_dir or data_dir() / "synthetic"
    _check_output_dir(out_dir)
    data = generate(seed=args.seed)
    for path in write_files(data, out_dir):
        frame = data.frames[next(k for k, v in FILE_NAMES.items() if v == path.name)]
        print(f"保存: {path}（{frame.height:,}行 × {frame.width}列）")


if __name__ == "__main__":
    main()
