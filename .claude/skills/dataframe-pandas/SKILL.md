---
name: dataframe-pandas
description: Use this when performing DataFrame operations — including loading, filtering, joining, aggregating, transforming, or reshaping tabular data with pandas.
---

# Skill: DataFrame Operations with Pandas

Use this skill for DataFrame operations.

## Default: Pandas

- Use **pandas** for all DataFrame work (pandas 3: Copy-on-Write is on, text columns use the `str` dtype).
- Do **not** add polars back as a dependency.
- Read only the columns you need (`pd.read_parquet(path, columns=[...], filters=[...])`, `pd.read_csv(usecols=...)`).
- For CSV files too large for memory, read in chunks (`util.csv_io.iter_csv_chunks`) and write with pyarrow's `ParquetWriter`.

## Conventions in This Project

- **Missing values:** NaN (NaT for datetimes). Use nullable dtypes (`Int64`, `Float64`, `boolean`) only where needed.
- **Sorting:** use `sort_values(..., kind="stable")` so equal keys keep their original order.
- **Group by:** use `groupby(keys, sort=False, dropna=False)` when the order of first appearance matters
  or missing values should form their own group.
- **Positions vs labels:** use `.iloc` for row positions (e.g. CV fold indices). Call `reset_index(drop=True)`
  after sorting or filtering when later code relies on positions.
- **Writing:** `to_csv(path, index=False)` / `to_parquet(path, index=False)`.
- **Datetimes:** parse with an explicit `format=` and keep `datetime64[us]` precision.

## Transformations

- Transformations should be reproducible and scriptable.
- Avoid manual, spreadsheet-like edits.
- Document data transformations with comments (in Japanese).
- Do not mutate input DataFrames in place inside functions; return a new DataFrame (`assign`, `copy`).

## Examples

### Read and Filter

```python
import pandas as pd

# 必要な列と期間だけを読む（Parquetなら pyarrow が不要な行を読み飛ばす）
events = pd.read_parquet(
    "data/raw/events.parquet",
    columns=["user_id", "event_type", "event_date"],
    filters=[("event_date", ">=", pd.Timestamp("2024-01-01"))],
)
```

### Group By and Aggregation

```python
# ユーザーごとのイベント数と最終イベント日を集計
summary = (
    events.groupby("user_id", sort=False)
    .agg(
        event_count=("event_type", "count"),
        last_event=("event_date", "max"),
    )
    .reset_index()
)
```

### Safe Join with Row Count Check

```python
users = pd.read_parquet("data/processed/users.parquet")
orders = pd.read_parquet("data/processed/orders.parquet")

# 右側のキーが一意であることを確かめてから結合する（validate で多対多を検出）
joined = users.merge(orders, on="user_id", how="left", validate="one_to_many")

# 結合後の行数を確認（ファンアウトの検出）
assert len(joined) >= len(users), "LEFT JOINで行が減少：joinキーを確認"
print(f"left={len(users)}, right={len(orders)}, joined={len(joined)}")
```
