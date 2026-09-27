"""feature_engineering.series_transform のテスト。"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from feature_engineering.numeric import LogTransformer
from feature_engineering.series_transform import (
    SERIES_TRANSFORM_KINDS,
    DifferenceTransformer,
    make_series_transformer,
)


def _series(n: int = 40, seed: int = 0) -> np.ndarray:
    """正の値を取る、トレンドと周期7の季節性を持つ系列。"""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 50 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 7) + rng.normal(size=n)


def _panel() -> pl.DataFrame:
    """2系列（A, B）が交互に並んだパネルデータ。"""
    a, b = _series(30, seed=1), _series(30, seed=2) * 3
    return pl.DataFrame(
        {
            "s": ["A", "B"] * 30,
            "t": np.repeat(np.arange(30), 2),
            "x": np.ravel(np.column_stack([a, b])),
            "y": np.ravel(np.column_stack([a + 100, b + 100])),
        }
    )


# --- 変換値の正しさ ---------------------------------------------------------------------


def test_diff_values_and_leading_nulls() -> None:
    X = pl.DataFrame({"x": [1.0, 3.0, 6.0, 10.0, 15.0]})
    out = DifferenceTransformer("x").fit_transform(X)
    assert out["x"].to_list() == [None, 2.0, 3.0, 4.0, 5.0]


def test_seasonal_diff_values() -> None:
    X = pl.DataFrame({"x": [1.0, 2.0, 3.0, 11.0, 22.0, 33.0]})
    out = DifferenceTransformer("x", periods=3).fit_transform(X)
    assert out["x"].to_list() == [None, None, None, 10.0, 20.0, 30.0]


def test_log_diff_values() -> None:
    X = pl.DataFrame({"x": [1.0, math.e, math.e**3]})
    out = DifferenceTransformer("x", log=True).fit_transform(X)
    assert out["x"].to_list()[1:] == pytest.approx([1.0, 2.0])


def test_diff_per_series_with_group_by() -> None:
    X = pl.DataFrame({"s": ["A", "B", "A", "B"], "x": [1.0, 10.0, 4.0, 30.0]})
    out = DifferenceTransformer("x", group_by="s").fit_transform(X)
    assert out["x"].to_list() == [None, None, 3.0, 20.0]


# --- 往復（5種類） ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", SERIES_TRANSFORM_KINDS)
def test_round_trip_single_series(kind: str) -> None:
    X = pl.DataFrame({"x": _series(), "y": _series(seed=5) + 10})
    t = make_series_transformer(kind, ["x", "y"], seasonal_period=7).fit(X)
    back = t.inverse_transform(t.transform(X))
    assert back["x"].to_list() == pytest.approx(X["x"].to_list())
    assert back["y"].to_list() == pytest.approx(X["y"].to_list())


@pytest.mark.parametrize("kind", SERIES_TRANSFORM_KINDS)
def test_round_trip_panel(kind: str) -> None:
    X = _panel()
    t = make_series_transformer(kind, ["x", "y"], seasonal_period=7, group_by="s").fit(X)
    back = t.inverse_transform(t.transform(X))
    assert back["x"].to_list() == pytest.approx(X["x"].to_list())
    assert back["y"].to_list() == pytest.approx(X["y"].to_list())


# --- 予測期間の復元 ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["diff", "log_diff", "seasonal_diff", "log_seasonal_diff"])
def test_inverse_continues_after_fit_data(kind: str) -> None:
    full = pl.DataFrame({"x": _series(40)})
    past, future = full.head(30), full.tail(10)
    t = make_series_transformer(kind, "x", seasonal_period=7).fit(past)
    # 全体を変換して後半の差分だけを取り出し（＝モデルが予測した差分の代わり）、元に戻す
    future_diff = t.transform(full).tail(10)
    by_fit = t.inverse_transform(future_diff, history="fit")
    by_history = t.inverse_transform(future_diff, history=past)
    assert by_fit["x"].to_list() == pytest.approx(future["x"].to_list())
    assert by_history["x"].to_list() == pytest.approx(future["x"].to_list())


def test_inverse_continues_after_explicit_history_per_series() -> None:
    X = _panel()
    past, future = X.filter(pl.col("t") < 20), X.filter(pl.col("t") >= 20)
    t = DifferenceTransformer(["x", "y"], periods=7, log=True, group_by="s").fit(X)
    future_diff = t.transform(X).filter(pl.col("t") >= 20)
    back = t.inverse_transform(future_diff, history=past)
    assert back["x"].to_list() == pytest.approx(future["x"].to_list())
    assert back["y"].to_list() == pytest.approx(future["y"].to_list())


# --- 異常系・境界 ------------------------------------------------------------------------


def test_missing_or_short_anchor_raises() -> None:
    X = pl.DataFrame({"s": ["A", "A", "A"], "x": [1.0, 2.0, 3.0]})
    t = DifferenceTransformer("x", periods=2, group_by="s").fit(X)
    with pytest.raises(ValueError, match="起点"):
        t.inverse_transform(pl.DataFrame({"s": ["B"], "x": [1.0]}), history="fit")
    with pytest.raises(ValueError, match="1 行しかありません"):
        t.inverse_transform(pl.DataFrame({"s": ["A"], "x": [1.0]}), history=X.head(1))
    with pytest.raises(ValueError, match="history"):
        t.inverse_transform(X, history="tail")  # type: ignore[arg-type]


def test_invalid_periods_raises() -> None:
    with pytest.raises(ValueError, match="periods"):
        DifferenceTransformer("x", periods=0).fit(pl.DataFrame({"x": [1.0]}))


def test_log_requires_positive_values_unless_offset() -> None:
    X = pl.DataFrame({"x": [0.0, 1.0, 3.0]})
    with pytest.raises(ValueError, match="offset"):
        DifferenceTransformer("x", log=True).fit(X)
    t = DifferenceTransformer("x", log=True, offset=1.0).fit(X)
    assert t.inverse_transform(t.transform(X))["x"].to_list() == pytest.approx([0.0, 1.0, 3.0])


def test_null_in_middle_propagates_to_same_phase() -> None:
    X = pl.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0, 6.0]})
    t = DifferenceTransformer("x", periods=2).fit(X)
    back = t.inverse_transform(t.transform(X))["x"].to_list()
    # 位相0（0, 2, 4番目）はnullの後が復元できない。位相1（1, 3, 5番目）は影響を受けない
    assert back[0] == 1.0 and back[2] is None and back[4] is None
    assert back[1::2] == pytest.approx([2.0, 4.0, 6.0])


# --- sklearn互換 ------------------------------------------------------------------------


def test_sklearn_compatibility() -> None:
    t = DifferenceTransformer("x", periods=3, log=True)
    assert t.get_params()["periods"] == 3
    cloned = clone(t.set_params(periods=7))
    assert cloned.periods == 7 and cloned is not t
    with pytest.raises(NotFittedError):
        t.transform(pl.DataFrame({"x": [1.0]}))


def test_pipeline_round_trip() -> None:
    # 対数 → 季節差分（= 対数季節差分と同じ）をPipelineでつなぎ、逆変換で元に戻す
    X = pl.DataFrame({"x": _series()})
    pipe = Pipeline(
        [("log", LogTransformer("x")), ("seasonal_diff", DifferenceTransformer("x", periods=7))]
    )
    transformed = pipe.fit_transform(X)
    expected = make_series_transformer("log_seasonal_diff", "x", seasonal_period=7).fit_transform(X)
    assert transformed["x"].to_list()[7:] == pytest.approx(expected["x"].to_list()[7:])
    back = pipe.inverse_transform(transformed)
    assert back["x"].to_list() == pytest.approx(X["x"].to_list())


# --- ファクトリ -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "cls", "periods", "log"),
    [
        ("log", LogTransformer, None, None),
        ("diff", DifferenceTransformer, 1, False),
        ("log_diff", DifferenceTransformer, 1, True),
        ("seasonal_diff", DifferenceTransformer, 12, False),
        ("log_seasonal_diff", DifferenceTransformer, 12, True),
    ],
)
def test_make_series_transformer(
    kind: str, cls: type, periods: int | None, log: bool | None
) -> None:
    t = make_series_transformer(kind, "x", seasonal_period=12)
    assert isinstance(t, cls)
    if isinstance(t, DifferenceTransformer):
        assert (t.periods, t.log) == (periods, log)


def test_make_series_transformer_unknown_kind() -> None:
    with pytest.raises(ValueError, match="未知"):
        make_series_transformer("box_cox", "x")
