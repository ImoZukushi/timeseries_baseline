"""旧形式の熱交換器ダミーデータ（`data/raw/dummy/` と同じ横持ちの6ファイル）を作るテスト用の部品。

旧形式を読む EDA・前処理（`scripts/eda_heat_exchanger_dummy.py`・
`scripts/preprocess_heat_exchanger_dummy.py`）のテストで、小さなデータを作るために使う。
`data/synthetic/` のダミーデータ（A部門の構成）は `scripts/generate_synthetic_heat_exchanger.py` が
生成する。以下は旧スクリプトの説明。

熱交換器10基の日次運転データ（ダミー）を生成し、`data/synthetic/train.csv` に保存するスクリプト。

原油の予熱トレインに並ぶ熱交換器（HX01〜HX10。番号が大きいほど高温側）を想定し、
2019-01-01〜2024-12-31（学習期間）と 2025-01-01〜2025-12-31（テスト期間）の日次データを、
物理的な関係を持たせて乱数で作る。テスト期間は学習期間の続きとして生成する。

生成の考え方:
    - 汚れ係数（m²·K/kW）は日ごとに「付着 − 除去」で増える（Ebert-Panchal 型の簡易モデル）。
      付着の速さは、原油C（重質）が多いほど・膜温度が高いほど・低温側流量が少ないほど大きく、
      薬剤（タイプA・タイプB）で抑えられる。除去は流量が多いほど大きい。
    - 汚れると総括伝熱係数 U = 1 / (1/U_clean + 汚れ係数) が下がり、熱交換量が減る
      （対向流の温度効率 ε-NTU から出口温度を計算）。低温側（原油側）の圧力損失は汚れで増える。
    - 低温側（原油）は HX01 → HX10 の順に直列に流れ、ある機器の低温側出口温度（から少しの放熱を
      引いた値）が次の機器の低温側入口温度になる。上流が汚れると下流の入口温度も下がる。
      停止中の機器はバイパスされ、原油は温度を変えずに次の機器へ流れる。
    - 汚れ係数が機器ごとの閾値を超えると、数日停止して洗浄し、小さな値に戻る（のこぎり形）。
      2021年10月には全基を止める定期修理（全基洗浄）がある。突発停止（トリップ）もある。
    - 停止中は流量 0、それ以外の測定値（圧力・汚れ係数・原油比率）は欠測。
    - センサーの欠測（ランダム・数日続くもの）と外れ値（スパイク・急落）を混ぜる。
      外れ値はセンサー値（圧力・流量）だけに入れ、目的変数（汚れ係数）には入れない。
    - 温度（高温側・低温側の入口・出口）は、汚れの進み方と原油の温度の受け渡しのために内部で
      計算するが、train.csv の列には含めない。
    - 原油比率は全基共通（同じ原油が予熱トレインを流れる）で、A + B + C = 1.0
      （小数3桁に丸め、C = 1 − A − B とする）。欠測・停止の日は3列とも欠測にする。

出力:
    - `data/synthetic/train.csv`: 横持ち（列 `日付`, `HX01_汚れ係数`, …, `HX10_薬剤種類`）
    - `data/synthetic/test.csv`: 2025-01-01〜2025-12-31 の運転データ（列は train.csv と同じ）。
      学習期間の最後の状態から続けて生成し、目的変数 `HXnn_汚れ係数` は null にする
    - `data/synthetic/test_answer.csv`: test.csv の目的変数の正解（列 `日付` と `HXnn_汚れ係数`）
    - `data/synthetic/events.csv`: 停止・洗浄の記録（列 `期間` に train / test。正解の確認用で、
      モデルの入力には使わない）
    - `data/synthetic/crude.csv`: 原油A〜Cの性状（密度・比重・API比重・粘度・硫黄分・金属分・
      アスファルテン分・炭化水素タイプ）。train.csv の `原油A比率` などと `原油` 列で対応する
    - `data/synthetic/chemical.csv`: 薬剤タイプA・タイプBの性状（有効成分濃度・pH・比重・溶解性）。
      train.csv の `HXnn_薬剤種類` と `薬剤種類` 列で結合できる

Usage:
    uv run python scripts/generate_synthetic_heat_exchanger.py
    uv run python scripts/generate_synthetic_heat_exchanger.py --seed 1 --output-dir data/synthetic
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from util.paths import data_dir  # noqa: E402

START = dt.date(2019, 1, 1)
END = dt.date(2024, 12, 31)  # 学習期間の終わり
TEST_END = dt.date(2025, 12, 31)  # テスト期間（END の翌日から）の終わり
N_EXCHANGERS = 10
CHEMICALS = ("なし", "タイプA", "タイプB")

# 列（機器ごと。横持ちでは `HXnn_<列名>` になる）
COLUMNS = (
    "汚れ係数",
    "入口圧力",
    "出口圧力",
    "高温側流量",
    "低温側流量",
    "原油A比率",
    "原油B比率",
    "原油C比率",
    "薬剤種類",
)
SENSOR_COLUMNS = COLUMNS[1:5]  # 外れ値・センサー欠測を入れる列（圧力・流量）
RATIO_COLUMNS = ("原油A比率", "原油B比率", "原油C比率")

# 比熱（kJ/kg/K）。高温側は製品油、低温側は原油
CP_HOT, CP_COLD = 2.6, 2.2
# 原油ごとの付着しやすさ（A: 軽質、B: 中質、C: 重質・アスファルテン分が多い）。
# crude.csv の性状（アスファルテン分・硫黄分・金属分が多いほど汚れやすい）と順序をそろえている
CRUDE_FOULING = np.array([0.6, 1.0, 2.2])

# 原油の性状（crude.csv）。API比重を基準にし、比重・密度はそこから計算して整合させる
CRUDE_NAMES = ("A", "B", "C")
CRUDE_API = (38.0, 29.0, 18.0)  # 軽質・中質・重質
CRUDE_VISCOSITY = (3.8, 14.5, 260.0)  # 動粘度（mm²/s、50℃）
CRUDE_SULFUR = (0.42, 1.85, 3.30)  # 全硫黄分（質量%）
CRUDE_METALS = ("ニッケル", "バナジウム・ニッケル", "バナジウム・ニッケル・鉄")
CRUDE_ASPHALTENE = (0.6, 2.8, 9.5)  # アスファルテン分（質量%）
CRUDE_HYDROCARBON_TYPE = ("パラフィン基", "混合基", "ナフテン基")
HYDROCARBON_TYPES = ("パラフィン基", "混合基", "ナフテン基")
METALS = ("バナジウム", "ニッケル", "鉄")
# 15℃の水の密度（g/cm³）。比重（15/15℃）= 原油の密度 / 水の密度
WATER_DENSITY_15C = 0.99910

# 薬剤（防汚剤）の性状（chemical.csv）。train.csv の 薬剤種類 のうち「なし」以外。
# タイプA は有効成分が濃く油によく溶けるため、付着を強く抑える（CHEMICAL_FACTOR 0.55 < 0.75）
CHEMICAL_TYPES = ("タイプA", "タイプB")
CHEMICAL_CONCENTRATION = (45.0, 30.0)  # 有効成分濃度（質量%）
CHEMICAL_PH = (7.8, 9.2)  # 水抽出液の pH（25℃）
CHEMICAL_GRAVITY = (0.92, 0.98)  # 比重（20/4℃）
CHEMICAL_SOLUBILITY = (35.0, 18.0)  # 灯油への溶解度（g/100g、25℃）
# 薬剤による付着の抑制（なし・タイプA・タイプB）
CHEMICAL_FACTOR = np.array([1.0, 0.55, 0.75])
# 定期修理（全基停止・全基洗浄）
TURNAROUND = (dt.date(2021, 10, 1), dt.date(2021, 10, 21))
# 停止・洗浄の記録（events.csv）の列
EVENT_SCHEMA = {
    "機器": pl.String,
    "開始日": pl.Date,
    "終了日": pl.Date,
    "種類": pl.String,
    "停止前の汚れ係数": pl.Float64,
}


@dataclass(frozen=True)
class Exchanger:
    """熱交換器1基の固有の値。"""

    name: str
    hot_inlet: float  # 高温側入口温度の基準（℃）
    hot_flow: float  # 高温側流量の基準（t/h）
    u_clean: float  # 汚れのない総括伝熱係数（kW/m²/K）
    area: float  # 伝熱面積（m²）
    inlet_pressure: float  # 入口圧力の基準（MPa）
    dp_clean: float  # 汚れのない圧力損失（MPa、基準流量のとき）
    fouling_scale: float  # 付着の速さの機器差
    threshold: float  # 洗浄する汚れ係数（m²·K/kW）
    residual: float  # 洗浄後に残る汚れ係数


def make_exchangers(rng: np.random.Generator, n: int = N_EXCHANGERS) -> list[Exchanger]:
    """予熱トレインの順に、だんだん高温になる熱交換器を作る。"""
    exchangers = []
    for i in range(n):
        exchangers.append(
            Exchanger(
                name=f"HX{i + 1:02d}",
                hot_inlet=180.0 + 15.0 * i + rng.normal(0, 3),
                hot_flow=rng.uniform(120, 250),
                u_clean=rng.uniform(0.45, 0.65),
                area=rng.uniform(150, 400),
                inlet_pressure=2.4 - 0.12 * i + rng.normal(0, 0.02),
                dp_clean=rng.uniform(0.05, 0.09),
                fouling_scale=float(np.exp(rng.normal(0, 0.15))),
                threshold=rng.uniform(1.0, 1.6),
                residual=rng.uniform(0.03, 0.08),
            )
        )
    return exchangers


def crude_slate(
    rng: np.random.Generator, n_days: int, current: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """全基共通の原油比率 (n_days, 3)。10〜40日ごとに目標の組成が変わり、数日かけて移る。

    Args:
        rng: 乱数生成器。
        n_days: 日数。
        current: 直前の期間の最後の組成（Noneなら乱数で決める）。

    Returns:
        (原油比率, 最後の組成)。最後の組成は、続きの期間を生成するときに渡す。
    """
    slate = np.empty((n_days, 3))
    if current is None:
        current = rng.dirichlet([4.0, 3.0, 1.5])
    day = 0
    while day < n_days:
        target = rng.dirichlet([4.0, 3.0, 1.5])
        length = int(rng.integers(10, 41))
        for _ in range(min(length, n_days - day)):
            # タンクの切り替えで数日かけて組成が移り、日々の小さな揺らぎもある
            current = current + (target - current) / 3.0
            noisy = np.clip(current + rng.normal(0, 0.01, 3), 0.001, None)
            slate[day] = noisy / noisy.sum()
            day += 1
    return slate, current


def chemical_campaigns(
    rng: np.random.Generator, n_days: int, carry: tuple[int, int] | None = None
) -> tuple[np.ndarray, tuple[int, int]]:
    """機器ごとの薬剤（0: なし、1: タイプA、2: タイプB）。60〜240日の期間ごとに切り替える。

    Args:
        rng: 乱数生成器。
        n_days: 日数。
        carry: 直前の期間から続く薬剤の (種類, 残り日数)（Noneなら新しく始める）。

    Returns:
        (薬剤の種類, 次の期間へ続く (種類, 残り日数))。
    """
    codes = np.empty(n_days, dtype=np.int64)
    day = 0
    code, remaining = carry if carry is not None else (0, 0)
    if remaining > 0:
        # 前の期間の薬剤を、残り日数だけ続ける
        day = min(remaining, n_days)
        codes[:day] = code
        remaining -= day
    while day < n_days:
        length = int(rng.integers(60, 241))
        code = int(rng.choice(3, p=[0.4, 0.35, 0.25]))
        codes[day : day + length] = code
        remaining = max(day + length - n_days, 0)
        day += length
    return codes, (code, remaining)


def counterflow_effectiveness(ntu: np.ndarray, ratio: np.ndarray) -> np.ndarray:
    """対向流熱交換器の温度効率 ε（ratio = Cmin / Cmax）。"""
    ratio = np.minimum(ratio, 0.999)  # ratio = 1 の特異点を避ける
    e = np.exp(-ntu * (1.0 - ratio))
    return (1.0 - e) / (1.0 - ratio * e)


@dataclass
class Simulation:
    """生成結果（1つの期間分）。

    Attributes:
        frame: 横持ちの運転データ（目的変数を消す前の内容）。
        events: 停止・洗浄の記録（機器・開始日・終了日・種類・停止前の汚れ係数）。
    """

    frame: pl.DataFrame
    events: pl.DataFrame


@dataclass
class TrainTest:
    """学習期間とテスト期間を続けて生成した結果。

    Attributes:
        train: 学習期間の生成結果（train.csv）。`simulate` と同じ内容。
        test: テスト期間の運転データ（test.csv。目的変数 `HXnn_汚れ係数` は null）。
        answer: テスト期間の目的変数（列 `日付` と `HXnn_汚れ係数`。正解の確認用）。
        events: 両期間の停止・洗浄の記録（列 `期間` に `train` / `test`）。
    """

    train: Simulation
    test: pl.DataFrame
    answer: pl.DataFrame
    events: pl.DataFrame


class _Simulator:
    """熱交換器の状態を持ち、期間を区切って続けて生成する。

    `run` を続けて呼ぶと、前の期間の最後の状態（汚れ係数・洗浄回数・洗浄中の残り日数・
    原油の組成・薬剤・処理量）から続きを生成する。乱数は1つの生成器を順に使うため、
    最初の期間の結果は「その期間だけを生成した場合」と同じになる
    （train.csv を変えずに、その続きとして test.csv を作れる）。
    """

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.exchangers: list[Exchanger] = []
        self.slate_current: np.ndarray | None = None
        self.chemical_carry: list[tuple[int, int] | None] = [None] * N_EXCHANGERS
        self.level = 300.0  # 処理量（t/h）
        self.fouling = np.zeros(N_EXCHANGERS)
        self.cleanings = np.zeros(N_EXCHANGERS, dtype=np.int64)
        self.cleaning_left = np.zeros(N_EXCHANGERS, dtype=np.int64)  # 洗浄のための停止の残り日数

    def run(self, start: dt.date, end: dt.date) -> Simulation:
        """start〜end（含む）の日次データを、前の期間の続きとして生成する。"""
        rng = self.rng
        dates = [start + dt.timedelta(days=d) for d in range((end - start).days + 1)]
        n_days, n_hx = len(dates), N_EXCHANGERS
        if not self.exchangers:
            self.exchangers = make_exchangers(rng, n_hx)
            self.fouling = np.array([hx.residual for hx in self.exchangers])
        exchangers = self.exchangers
        slate, self.slate_current = crude_slate(rng, n_days, self.slate_current)
        chemical_list = []
        for j in range(n_hx):
            codes, self.chemical_carry[j] = chemical_campaigns(rng, n_days, self.chemical_carry[j])
            chemical_list.append(codes)
        chemicals = np.stack(chemical_list, axis=1)
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        season = np.sin(2 * np.pi * (day_of_year - 110) / 365.25)  # 夏に高く冬に低い

        # 処理量（低温側＝原油の流量）は全基共通。需要に応じてゆっくり変わる
        throughput = np.empty(n_days)
        for d in range(n_days):
            self.level += 0.05 * (300.0 - self.level) + rng.normal(0, 3.0)
            throughput[d] = self.level

        # 突発停止（トリップ）: 機器ごとに年1回程度、1〜3日。洗浄はしない
        trips = np.zeros((n_days, n_hx), dtype=bool)
        events: list[dict[str, object]] = []
        for j, hx in enumerate(exchangers):
            for _ in range(int(rng.poisson(n_days / 365.25))):
                first = int(rng.integers(0, n_days - 3))
                length = int(rng.integers(1, 4))
                trips[first : first + length, j] = True
                events.append(
                    {
                        "機器": hx.name,
                        "開始日": dates[first],
                        "終了日": dates[first + length - 1],
                        "種類": "突発停止",
                        "停止前の汚れ係数": None,
                    }
                )
        turnaround = np.array([TURNAROUND[0] <= d <= TURNAROUND[1] for d in dates])

        columns: dict[str, np.ndarray] = {
            name: np.full((n_days, n_hx), np.nan) for name in COLUMNS if name != "薬剤種類"
        }
        fouling, cleanings, cleaning_left = self.fouling, self.cleanings, self.cleaning_left

        for d in range(n_days):
            # 原油（低温側）はタンクから HX01 に入る。外気の影響で夏に高く冬に低い
            stream_temp = 30.0 + 6.0 * season[d] + rng.normal(0, 1.0)
            for j, hx in enumerate(exchangers):
                stopped = turnaround[d] or trips[d, j] or cleaning_left[j] > 0
                if turnaround[d] and (d == 0 or not turnaround[d - 1]):
                    events.append(
                        {
                            "機器": hx.name,
                            "開始日": TURNAROUND[0],
                            "終了日": TURNAROUND[1],
                            "種類": "定期修理（洗浄）",
                            "停止前の汚れ係数": round(float(fouling[j]), 4),
                        }
                    )
                    cleanings[j] += 1
                    fouling[j] = hx.residual + 0.01 * cleanings[j]
                if stopped:
                    # 停止中はバイパスされ、原油は温度を変えずに次の機器へ流れる
                    cleaning_left[j] = max(cleaning_left[j] - 1, 0)
                    columns["高温側流量"][d, j] = 0.0
                    columns["低温側流量"][d, j] = 0.0
                    continue

                # --- 運転条件 ---
                hot_in = hx.hot_inlet + 3.0 * season[d] + rng.normal(0, 1.5)
                cold_in = stream_temp
                hot_flow = hx.hot_flow * (1 + rng.normal(0, 0.03))
                cold_flow = throughput[d] * (1 + rng.normal(0, 0.01))

                # --- 伝熱（汚れで U が下がり、熱交換量が減る） ---
                u = 1.0 / (1.0 / hx.u_clean + fouling[j])
                c_hot = hot_flow * 1000 / 3600 * CP_HOT  # kW/K
                c_cold = cold_flow * 1000 / 3600 * CP_COLD
                c_min, c_max = min(c_hot, c_cold), max(c_hot, c_cold)
                eff = counterflow_effectiveness(
                    np.array(u * hx.area / c_min), np.array(c_min / c_max)
                )
                duty = float(eff) * c_min * (hot_in - cold_in)
                hot_out = hot_in - duty / c_hot
                cold_out = cold_in + duty / c_cold
                # 次の機器へ（配管での放熱で少し下がる）
                stream_temp = cold_out - rng.uniform(0.5, 2.0)

                # --- 圧力（低温側。汚れで流路が狭まり圧力損失が増える） ---
                p_in = hx.inlet_pressure + 0.002 * (cold_flow - 300.0) + rng.normal(0, 0.01)
                dp = hx.dp_clean * (cold_flow / 300.0) ** 1.8 * (1.0 + 0.9 * fouling[j])
                p_out = p_in - dp

                # 温度は汚れの進み方（膜温度）と次の機器への原油の温度に使うが、出力しない
                values = {
                    "汚れ係数": fouling[j],
                    "入口圧力": p_in,
                    "出口圧力": p_out,
                    "高温側流量": hot_flow,
                    "低温側流量": cold_flow,
                }
                for name, value in values.items():
                    columns[name][d, j] = value
                for k, name in enumerate(RATIO_COLUMNS):
                    columns[name][d, j] = slate[d, k]

                # --- 翌日の汚れ係数（付着 − 除去） ---
                film_temp = 0.5 * (0.5 * (hot_in + hot_out) + 0.5 * (cold_in + cold_out))
                velocity = cold_flow / 300.0
                deposition = (
                    0.0042
                    * hx.fouling_scale
                    * float(slate[d] @ CRUDE_FOULING)
                    * CHEMICAL_FACTOR[chemicals[d, j]]
                    * velocity**-0.8
                    * np.exp(0.012 * (film_temp - 200.0))
                    * np.exp(rng.normal(0, 0.3))
                )
                removal = 0.001 * velocity**0.8 * fouling[j]
                fouling[j] = max(fouling[j] + deposition - removal, 0.0)

                # 閾値を超えたら、翌日から3〜7日止めて洗浄する（期間の最終日は次の期間で判定）
                if fouling[j] > hx.threshold and d + 1 < n_days:
                    length = int(rng.integers(3, 8))
                    cleaning_left[j] = length
                    events.append(
                        {
                            "機器": hx.name,
                            "開始日": dates[d + 1],
                            "終了日": dates[min(d + length, n_days - 1)],
                            "種類": "洗浄停止",
                            "停止前の汚れ係数": round(float(fouling[j]), 4),
                        }
                    )
                    cleanings[j] += 1
                    # 洗浄しきれない汚れが、洗浄のたびに少しずつ残る
                    fouling[j] = hx.residual + 0.01 * cleanings[j]

        observed = _add_measurement_noise(rng, columns)
        _add_missing_and_outliers(rng, observed)
        return Simulation(
            frame=_to_wide_frame(dates, exchangers, observed, chemicals),
            # 期間が短く停止が1件も無くても、同じ列を持つ表にする
            events=pl.DataFrame(events, schema=EVENT_SCHEMA).sort("開始日", "機器"),
        )


def simulate(seed: int = 42, start: dt.date = START, end: dt.date = END) -> Simulation:
    """熱交換器10基の日次運転データを生成する（1つの期間）。

    Args:
        seed: 乱数シード（同じ値なら同じデータになる）。
        start: 開始日。
        end: 終了日（含む）。

    Returns:
        生成結果（横持ちの DataFrame と停止・洗浄の記録）。
    """
    return _Simulator(seed).run(start, end)


def simulate_train_test(
    seed: int = 42,
    start: dt.date = START,
    train_end: dt.date = END,
    test_end: dt.date = TEST_END,
) -> TrainTest:
    """学習期間と、その続きのテスト期間を生成する。

    テスト期間は学習期間の最後の状態（汚れ係数・洗浄の状況・原油の組成・薬剤・処理量）から
    続けて生成するため、運転データは学習期間と途切れずにつながる。学習期間の結果は
    `simulate(seed, start, train_end)` と同じ。

    Args:
        seed: 乱数シード。
        start: 学習期間の開始日。
        train_end: 学習期間の終了日（含む）。テスト期間はその翌日から。
        test_end: テスト期間の終了日（含む）。

    Returns:
        学習・テストの生成結果。テストの運転データは目的変数（`HXnn_汚れ係数`）を null にし、
        正解は `answer` に分ける。
    """
    simulator = _Simulator(seed)
    train = simulator.run(start, train_end)
    test_full = simulator.run(train_end + dt.timedelta(days=1), test_end)
    target_columns = [c for c in test_full.frame.columns if c.endswith("_汚れ係数")]
    test = test_full.frame.with_columns(
        pl.lit(None, dtype=pl.Float64).alias(c) for c in target_columns
    )
    events = pl.concat(
        [
            train.events.with_columns(pl.lit("train").alias("期間")),
            test_full.events.with_columns(pl.lit("test").alias("期間")),
        ]
    )
    return TrainTest(
        train=train,
        test=test,
        answer=test_full.frame.select("日付", *target_columns),
        events=events,
    )


def _add_measurement_noise(
    rng: np.random.Generator, columns: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """測定の誤差を加え、桁を丸める（停止中の流量 0 はそのまま）。"""
    noise = {
        "汚れ係数": 0.02,
        "入口圧力": 0.005,
        "出口圧力": 0.005,
    }
    digits = {"汚れ係数": 4, "入口圧力": 3, "出口圧力": 3}
    out = {}
    for name, values in columns.items():
        v = values.copy()
        if name in noise:
            v = v + rng.normal(0, noise[name], v.shape)
        elif name in ("高温側流量", "低温側流量"):
            v = np.where(v > 0, v * (1 + rng.normal(0, 0.005, v.shape)), v)
        if name == "汚れ係数":
            v = np.maximum(v, 0.0)
        if name not in RATIO_COLUMNS:
            v = np.round(v, digits.get(name, 1))
        out[name] = v
    # 原油比率: 小数3桁に丸め、C = 1 − A − B で合計を 1.0 にそろえる
    a = np.round(out["原油A比率"], 3)
    b = np.round(out["原油B比率"], 3)
    out["原油A比率"], out["原油B比率"] = a, b
    out["原油C比率"] = np.round(1.0 - a - b, 3)
    return out


def _add_missing_and_outliers(rng: np.random.Generator, columns: dict[str, np.ndarray]) -> None:
    """センサーの欠測（ランダム・数日続くもの）と外れ値を入れる（停止中の行は除く）。"""
    running = columns["低温側流量"] > 0
    n_days, n_hx = running.shape
    for name in (*SENSOR_COLUMNS, "汚れ係数"):
        v = columns[name]
        # ランダムな欠測（約1%）
        v[(rng.random(v.shape) < 0.01) & running] = np.nan
    for j in range(n_hx):
        # 数日続くセンサーの欠測（年1回程度、ランダムな1列、2〜10日）
        for _ in range(int(rng.poisson(n_days / 365.25))):
            name = SENSOR_COLUMNS[int(rng.integers(len(SENSOR_COLUMNS)))]
            first = int(rng.integers(0, n_days - 10))
            columns[name][first : first + int(rng.integers(2, 11)), j] = np.nan
    for name in SENSOR_COLUMNS:
        v = columns[name]
        # 外れ値（約0.2%）: 値が跳ね上がるスパイクと、落ち込む急落
        mask = (rng.random(v.shape) < 0.002) & running & ~np.isnan(v)
        factor = np.where(
            rng.random(v.shape) < 0.5,
            rng.uniform(1.3, 2.0, v.shape),
            rng.uniform(0.2, 0.6, v.shape),
        )
        digits = 3 if "圧力" in name else 1
        v[mask] = np.round(v[mask] * factor[mask], digits)
    # 原油比率の欠測（約0.5%）は3列そろって欠ける（合計 1.0 の関係を崩さない）
    missing = (rng.random((n_days, n_hx)) < 0.005) & running
    for name in RATIO_COLUMNS:
        columns[name][missing] = np.nan


def _to_wide_frame(
    dates: list[dt.date],
    exchangers: list[Exchanger],
    columns: dict[str, np.ndarray],
    chemicals: np.ndarray,
) -> pl.DataFrame:
    """横持ち（`日付`, `HX01_汚れ係数`, …）の DataFrame にする。"""
    data: dict[str, pl.Series] = {"日付": pl.Series("日付", dates, dtype=pl.Date)}
    for j, hx in enumerate(exchangers):
        for name in COLUMNS:
            col = f"{hx.name}_{name}"
            if name == "薬剤種類":
                labels = [CHEMICALS[c] for c in chemicals[:, j]]
                data[col] = pl.Series(col, labels, dtype=pl.Categorical)
            else:
                data[col] = pl.Series(col, columns[name][:, j], dtype=pl.Float64).fill_nan(None)
    return pl.DataFrame(data)


def crude_properties() -> pl.DataFrame:
    """原油A〜Cの性状の表（crude.csv の内容）。

    API比重から比重（15/15℃）= 141.5 / (API + 131.5)、密度（15℃）= 比重 × 水の密度 を計算する。
    重質の原油ほど、粘度・硫黄分・金属分・アスファルテン分が多い（train.csv で汚れやすい順と同じ）。

    Returns:
        列 `原油`, `密度`, `比重`, `API比重`, `粘度`, `全硫黄分`, `金属分`, `アスファルテン分`,
        `炭化水素タイプ`（categorical）の3行の表。
    """
    gravity = [141.5 / (api + 131.5) for api in CRUDE_API]
    return pl.DataFrame(
        {
            "原油": list(CRUDE_NAMES),
            "密度": [round(sg * WATER_DENSITY_15C, 4) for sg in gravity],
            "比重": [round(sg, 4) for sg in gravity],
            "API比重": list(CRUDE_API),
            "粘度": list(CRUDE_VISCOSITY),
            "全硫黄分": list(CRUDE_SULFUR),
            "金属分": list(CRUDE_METALS),
            "アスファルテン分": list(CRUDE_ASPHALTENE),
            "炭化水素タイプ": pl.Series(
                list(CRUDE_HYDROCARBON_TYPE), dtype=pl.Enum(list(HYDROCARBON_TYPES))
            ),
        }
    )


def chemical_properties() -> pl.DataFrame:
    """薬剤（防汚剤）タイプA・タイプBの性状の表（chemical.csv の内容）。

    `薬剤種類` 列の値は train.csv の `HXnn_薬剤種類` と同じで、そのまま結合できる
    （「なし」の行は無い）。付着を強く抑える薬剤（タイプA）ほど、有効成分が濃く油によく溶ける。

    Returns:
        列 `薬剤種類`, `有効成分濃度`, `ph`, `比重`, `溶解性` の2行の表。
    """
    return pl.DataFrame(
        {
            "薬剤種類": list(CHEMICAL_TYPES),
            "有効成分濃度": list(CHEMICAL_CONCENTRATION),
            "ph": list(CHEMICAL_PH),
            "比重": list(CHEMICAL_GRAVITY),
            "溶解性": list(CHEMICAL_SOLUBILITY),
        }
    )


def main(argv: list[str] | None = None) -> None:
    """ダミーデータを生成して保存する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="乱数シード（既定: 42）")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="保存先ディレクトリ（既定: data/synthetic）",
    )
    args = parser.parse_args(argv)

    out_dir = args.output_dir or data_dir() / "synthetic"
    if out_dir.resolve().is_relative_to(
        (data_dir() / "raw").resolve()
    ) or out_dir.resolve().is_relative_to((data_dir() / "external").resolve()):
        raise ValueError(f"data/raw・data/external には書き込めません: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = simulate_train_test(seed=args.seed)
    outputs = {
        "train.csv": result.train.frame,
        "test.csv": result.test,
        "test_answer.csv": result.answer,
        "events.csv": result.events,
    }
    for filename, frame in outputs.items():
        path = out_dir / filename
        frame.write_csv(path)
        print(f"保存: {path}（{frame.height}行 × {frame.width}列）")
    crude = crude_properties()
    crude_path = out_dir / "crude.csv"
    crude.write_csv(crude_path)
    print(f"保存: {crude_path}（{crude.height}行 × {crude.width}列）")
    chemical = chemical_properties()
    chemical_path = out_dir / "chemical.csv"
    chemical.write_csv(chemical_path)
    print(f"保存: {chemical_path}（{chemical.height}行 × {chemical.width}列）")


if __name__ == "__main__":
    main()
