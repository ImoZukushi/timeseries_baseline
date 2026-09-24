"""modeling.cv のテスト。"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from modeling.config import CVConfig
from modeling.cv import TimeCutoffSplit, folds_to_ids, make_folds


def _frame(n: int = 100) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": np.arange(n, dtype=float),
            "ts": pl.datetime_range(
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 1) + dt.timedelta(days=n - 1),
                interval="1d",
                eager=True,
            ),
        }
    )


@pytest.mark.parametrize(
    "cfg",
    [
        CVConfig(method="kfold", n_splits=5),
        CVConfig(method="stratified", n_splits=5),
        CVConfig(method="group", n_splits=5),
        CVConfig(method="stratified_group", n_splits=4),
    ],
)
def test_non_temporal_folds_partition_all_rows(cfg: CVConfig) -> None:
    n = 100
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    groups = rng.integers(0, 20, n)
    folds = make_folds(cfg, _frame(n), y, groups)
    assert len(folds) == cfg.n_splits
    for train_idx, valid_idx in folds:
        # 学習と検証が重ならない
        assert set(train_idx).isdisjoint(valid_idx)
        assert len(train_idx) + len(valid_idx) == n
    # 全行がちょうど1回ずつ検証データになる
    ids = folds_to_ids(folds, n)
    assert (ids >= 0).all()
    assert sorted(np.concatenate([va for _, va in folds]).tolist()) == list(range(n))


@pytest.mark.parametrize("method", ["group", "stratified_group"])
def test_group_folds_do_not_split_groups(method: str) -> None:
    n = 120
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, n)
    groups = rng.integers(0, 15, n)
    folds = make_folds(CVConfig(method=method, n_splits=3), _frame(n), y, groups)
    for train_idx, valid_idx in folds:
        assert set(groups[train_idx]).isdisjoint(groups[valid_idx])


def test_kfold_is_reproducible_with_seed() -> None:
    cfg = CVConfig(method="kfold", n_splits=4, seed=7)
    y = np.zeros(50)
    a = make_folds(cfg, _frame(50), y)
    b = make_folds(cfg, _frame(50), y)
    assert all((x[1] == z[1]).all() for x, z in zip(a, b, strict=True))


def test_time_series_split_respects_order_and_gap() -> None:
    cfg = CVConfig(method="time_series", n_splits=4, gap=3)
    folds = make_folds(cfg, _frame(100), np.zeros(100))
    for train_idx, valid_idx in folds:
        # 検証期間は常に学習期間より後で、間にgap行空いている
        assert train_idx.max() + cfg.gap < valid_idx.min()
    # 最初の検証期間より前の行はどのfoldの検証にも入らない
    ids = folds_to_ids(folds, 100)
    assert (ids[: folds[0][1].min()] == -1).all()


def test_sliding_window_limits_train_size() -> None:
    cfg = CVConfig(method="sliding_window", n_splits=4, max_train_size=20)
    folds = make_folds(cfg, _frame(100), np.zeros(100))
    for train_idx, valid_idx in folds:
        assert len(train_idx) <= 20
        assert train_idx.max() < valid_idx.min()


def test_sliding_window_requires_max_train_size() -> None:
    with pytest.raises(ValueError):
        CVConfig(method="sliding_window")


def test_time_cutoff_split_uses_explicit_boundaries() -> None:
    frame = _frame(30)  # 2024-01-01 .. 2024-01-30
    splitter = TimeCutoffSplit("ts", ["2024-01-11", "2024-01-21"], valid_end="2024-01-26")
    folds = list(splitter.split(frame))
    assert splitter.get_n_splits() == 2
    (tr0, va0), (tr1, va1) = folds
    assert tr0.tolist() == list(range(10))
    assert va0.tolist() == list(range(10, 20))
    assert tr1.tolist() == list(range(20))
    assert va1.tolist() == list(range(20, 25))  # valid_endの日は含まない


def test_time_cutoff_split_accepts_string_time_column() -> None:
    frame = _frame(10).with_columns(pl.col("ts").dt.strftime("%Y-%m-%d %H:%M:%S"))
    folds = list(TimeCutoffSplit("ts", ["2024-01-06"]).split(frame))
    assert folds[0][1].tolist() == list(range(5, 10))


def test_time_cutoff_split_rejects_unsorted_cutoffs() -> None:
    with pytest.raises(ValueError):
        list(TimeCutoffSplit("ts", ["2024-01-20", "2024-01-10"]).split(_frame(30)))


def test_time_cutoff_split_rejects_empty_period() -> None:
    with pytest.raises(ValueError):
        list(TimeCutoffSplit("ts", ["2023-01-01"]).split(_frame(30)))


def test_time_cutoff_via_make_folds_uses_time_column_argument() -> None:
    cfg = CVConfig(method="time_cutoff", cutoffs=["2024-01-11"])
    folds = make_folds(cfg, _frame(30), np.zeros(30), time_column="ts")
    assert folds[0][0].tolist() == list(range(10))
