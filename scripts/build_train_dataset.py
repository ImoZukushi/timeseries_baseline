"""train.csv に diagram.csv・stop_station_location.csv を紐付けて訓練データを作成するスクリプト。

data/raw/train.csv の「列車番号」「停車駅名」をキーとして、data/raw/diagram.csv
（列車番号ごとの各駅発車時刻表、駅×列車番号のワイド形式）から該当する時刻を引き当て、
train.csv に「発車時刻」列として付与する。さらに data/raw/stop_station_location.csv
（駅ごとのキロ程・緯度・経度）を「停車駅名」（train側）と「停車場名」
（stop_station_location側）をキーに左結合し、駅の位置情報も付与する。

前提・仕様:
    - diagram.csv・stop_station_location.csv の駅名列はいずれも「停車場名」という列名だが、
      train.csvの「停車駅名」と同じ意味・表記のため、この列をキーとして結合する。
    - diagram.csv 中の "↓" は「時刻表示なし（前駅からの通過等で個別時刻が示されない）」
      を意味するため、発車時刻としては欠損（null）として扱う。
    - 発車時刻は元データの文字列表現（例: "6:13"）のままtrain.csvに格納する
      （時刻型への変換は行わない）。
    - stop_station_location.csv は「停車場名」がユニークなマスタテーブルであるため、
      左結合してもtrain.csv側の行数は増えない。
    - いずれの結合もtrain.csvの行数を変えない left join とし、対応する値が無い場合は
      該当列をnullのままにする。

入力: data/raw/train.csv, data/raw/diagram.csv, data/raw/stop_station_location.csv
      （いずれも不変・変更しない）
出力: data/processed/train_with_departure_time.csv

Usage:
    uv run python scripts/build_train_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl

from util.csv_io import read_csv_auto
from util.paths import data_dir, ensure_parent_dir


def build_departure_time_table(diagram_df: pl.DataFrame) -> pl.DataFrame:
    """diagram.csv（駅×列車番号のワイド形式）を (列車番号, 停車駅名, 発車時刻) の縦持ちに変換する。

    Args:
        diagram_df: diagram.csv を読み込んだDataFrame（1列目が駅名、以降が列車番号ごとの時刻）。

    Returns:
        列車番号・停車駅名をキーとした発車時刻の対応表。"↓"（時刻表示なし）はnullにする。
    """
    long_df = diagram_df.unpivot(index="停車場名", variable_name="列車番号", value_name="発車時刻")
    return long_df.with_columns(
        pl.when(pl.col("発車時刻") == "↓")
        .then(None)
        .otherwise(pl.col("発車時刻"))
        .alias("発車時刻")
    ).rename({"停車場名": "停車駅名"})


def join_stop_station_location(train_df: pl.DataFrame, location_df: pl.DataFrame) -> pl.DataFrame:
    """train.csvにstop_station_location.csvの駅位置情報（キロ程・緯度・経度）を左結合する。

    train_dfの「停車駅名」とlocation_dfの「停車場名」をキーとする。location_dfは
    「停車場名」がユニークなマスタテーブルであることを前提としており、train_df側の
    行数は変化しない。

    Args:
        train_df: train.csvを読み込んだDataFrame（「停車駅名」列を含む）。
        location_df: stop_station_location.csvを読み込んだDataFrame（「停車場名」列を含む）。

    Returns:
        駅位置情報を付与したDataFrame。
    """
    return train_df.join(location_df, left_on="停車駅名", right_on="停車場名", how="left")


def main() -> None:
    """train.csvにdiagram.csv・stop_station_location.csvを紐付けたデータセットを作成する。"""
    raw_dir = data_dir() / "raw"
    train_df = read_csv_auto(raw_dir / "train.csv")
    diagram_df = read_csv_auto(raw_dir / "diagram.csv")
    location_df = read_csv_auto(raw_dir / "stop_station_location.csv")

    departure_times = build_departure_time_table(diagram_df)

    n_rows_before = train_df.height
    result = train_df.join(departure_times, on=["列車番号", "停車駅名"], how="left")

    # left joinで行数が変化していないか（キーの重複によるファンアウトが無いか）を確認する
    assert result.height == n_rows_before, (
        f"発車時刻のjoinで行数が変化しました: {n_rows_before} -> {result.height}"
    )

    n_missing = result["発車時刻"].null_count()
    print(f"train.csv 行数: {n_rows_before:,}")
    print(f"発車時刻が紐付いた行数: {n_rows_before - n_missing:,}")
    print(f"発車時刻が紐付かなかった行数: {n_missing:,}")

    result = join_stop_station_location(result, location_df)

    assert result.height == n_rows_before, (
        f"駅位置情報のjoinで行数が変化しました: {n_rows_before} -> {result.height}"
    )

    n_missing_location = result["キロ程"].null_count()
    print(f"駅位置情報が紐付いた行数: {n_rows_before - n_missing_location:,}")
    print(f"駅位置情報が紐付かなかった行数: {n_missing_location:,}")

    output_path = ensure_parent_dir(data_dir() / "processed" / "train_with_departure_time.csv")
    result.write_csv(output_path)
    print(f"\n保存先: {output_path}")


if __name__ == "__main__":
    main()
