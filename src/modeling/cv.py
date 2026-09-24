"""CV分割の生成。

`make_cv` が返すsplitterは、すべてsklearnの `split(X, y, groups)` / `get_n_splits()` 規約に
従うため、`sklearn.model_selection.cross_validate` 等にもそのまま渡せる。
本パッケージの学習ループでは、分割を一度だけ計算した `list[(train_idx, valid_idx)]`
（`make_folds` の戻り値）を使い回す。これによりチューニングの各試行や
アンサンブル対象の各実験で完全に同じ分割を使うことが保証される。
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from modeling.config import CVConfig

Fold = tuple[np.ndarray, np.ndarray]


class TimeCutoffSplit(BaseCrossValidator):
    """日時のカットオフで分割する時系列CV。

    `cutoffs = [c1, c2, ..., cK]` のとき、第i foldは
    学習: `time < c_i`、検証: `c_i <= time < c_{i+1}`（最後のfoldは `valid_end` まで、
    未指定ならデータの最後まで）とする。本番で「ある時点までのデータで学習し、
    その後の期間を予測する」状況を再現するためのsplitter。

    Args:
        time_column: 時刻列名（`split` に渡す `X` から読む）。
        cutoffs: 検証期間の開始時刻（ISO形式文字列、または datetime/date）のリスト。昇順であること。
        valid_end: 最後の検証期間の終了時刻（この時刻は含まない）。
    """

    def __init__(
        self,
        time_column: str,
        cutoffs: Sequence[str | dt.datetime | dt.date],
        valid_end: str | dt.datetime | dt.date | None = None,
    ) -> None:
        self.time_column = time_column
        self.cutoffs = cutoffs
        self.valid_end = valid_end

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None) -> int:
        """fold数（カットオフの数）を返す。"""
        return len(self.cutoffs)

    def split(self, X: Any, y: Any = None, groups: Any = None) -> Iterator[Fold]:
        """学習・検証のインデックスを順に返す。

        Raises:
            ValueError: カットオフが昇順でない場合、または学習・検証期間が空になる場合。
        """
        times = _as_datetime_series(X[self.time_column])
        bounds = [_parse_time(c) for c in self.cutoffs]
        if bounds != sorted(bounds):
            raise ValueError("cutoffs は昇順で指定してください")
        end = _parse_time(self.valid_end) if self.valid_end is not None else None
        uppers: list[dt.datetime | None] = [*bounds[1:], end]
        for lower, upper in zip(bounds, uppers, strict=True):
            train_mask = (times < lower).fill_null(False)
            valid_mask = times >= lower
            if upper is not None:
                valid_mask = valid_mask & (times < upper)
            valid_mask = valid_mask.fill_null(False)
            train_idx = np.flatnonzero(train_mask.to_numpy())
            valid_idx = np.flatnonzero(valid_mask.to_numpy())
            if len(train_idx) == 0 or len(valid_idx) == 0:
                raise ValueError(f"カットオフ {lower} で学習または検証期間が空になります")
            yield train_idx, valid_idx

    def _iter_test_indices(self, X: Any = None, y: Any = None, groups: Any = None) -> Any:
        # splitを直接オーバーライドしているため使わない（BaseCrossValidatorの抽象メソッド）
        raise NotImplementedError


def _parse_time(value: str | dt.datetime | dt.date) -> dt.datetime:
    """カットオフ指定をdatetimeに変換する。"""
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime.combine(value, dt.time())
    return dt.datetime.fromisoformat(value)


def _as_datetime_series(values: Any) -> pl.Series:
    """時刻列をpolarsのDatetime Seriesに変換する（文字列・Date型も受け付ける）。"""
    series = values if isinstance(values, pl.Series) else pl.Series(values)
    if series.dtype == pl.String:
        return series.str.to_datetime()
    if series.dtype == pl.Date:
        return series.cast(pl.Datetime)
    return series


def make_cv(cfg: CVConfig, time_column: str | None = None) -> BaseCrossValidator | Any:
    """設定からsklearn互換のsplitterを作る。

    Args:
        cfg: CV設定。
        time_column: `time_cutoff` で使う時刻列（`cfg.time_column` が優先）。

    Returns:
        sklearn互換のsplitter。
    """
    method = cfg.method
    seed = cfg.seed if cfg.shuffle else None
    if method == "kfold":
        return KFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=seed)
    if method == "stratified":
        return StratifiedKFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=seed)
    if method == "group":
        # GroupKFoldのshuffleはsklearn 1.6+で利用可能
        return GroupKFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=seed)
    if method == "stratified_group":
        return StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=cfg.shuffle, random_state=seed)
    if method in ("time_series", "sliding_window"):
        # sliding_window は学習期間の長さを max_train_size で固定したTimeSeriesSplit
        return TimeSeriesSplit(
            n_splits=cfg.n_splits,
            gap=cfg.gap,
            max_train_size=cfg.max_train_size,
            test_size=cfg.test_size,
        )
    if method == "time_cutoff":
        column = cfg.time_column or time_column
        if column is None or cfg.cutoffs is None:
            raise ValueError("time_cutoff には時刻列と cutoffs が必要です")
        return TimeCutoffSplit(column, cfg.cutoffs, cfg.valid_end)
    raise ValueError(f"未知のCV方法です: {method}")


def make_folds(
    cfg: CVConfig,
    frame: pl.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    time_column: str | None = None,
) -> list[Fold]:
    """CV分割を計算してインデックスのリストで返す。

    Args:
        cfg: CV設定。
        frame: 学習データ全体（`time_cutoff` の時刻列を含む）。行数の基準にもなる。
        y: 目的変数（stratified系で使用）。
        groups: グループ（group系で使用）。
        time_column: `time_cutoff` で使う時刻列。

    Returns:
        `(train_idx, valid_idx)` のリスト。
    """
    splitter = make_cv(cfg, time_column=time_column)
    # group系以外のsplitterにgroupsを渡すとsklearnが警告を出すため、必要な場合のみ渡す
    if cfg.method not in ("group", "stratified_group"):
        groups = None
    return [
        (np.asarray(tr, dtype=np.int64), np.asarray(va, dtype=np.int64))
        for tr, va in splitter.split(frame, y, groups)
    ]


def folds_to_ids(folds: Sequence[Fold], n_samples: int) -> np.ndarray:
    """各行がどのfoldの検証データかを表す配列を返す（どのfoldにも入らない行は-1）。

    アンサンブル時に「全実験で同じCV分割か」を照合するためにも使う。
    """
    fold_ids = np.full(n_samples, -1, dtype=np.int64)
    for i, (_, valid_idx) in enumerate(folds):
        fold_ids[valid_idx] = i
    return fold_ids
