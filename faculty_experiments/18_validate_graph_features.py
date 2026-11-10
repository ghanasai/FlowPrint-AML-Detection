
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA = BASE / "data"

print("STEP 16C: GRAPH FEATURE VALIDATION")
print("=" * 50)

source = pd.read_csv(
    ROOT / "AMLNet_v2_transactions.csv",
    usecols=["nameOrig", "nameDest"]
)
source["original_row_id"] = np.arange(len(source))

timestamps = pd.read_csv(
    DATA / "historical_features.csv",
    usecols=["original_row_id", "timestamp"],
    parse_dates=["timestamp"]
)

graph = pd.read_csv(DATA / "graph_features.csv")

df = (
    source.merge(timestamps, on="original_row_id",
                 validate="one_to_one")
    .merge(graph, on="original_row_id",
           validate="one_to_one")
)

assert len(df) == 1_090_000
assert df["original_row_id"].is_unique

FEATURES = [
    "sender_out_degree",
    "sender_in_degree",
    "receiver_in_degree",
    "receiver_out_degree",
    "pair_previous_count"
]

assert not df[FEATURES].isna().any().any()

# Recalculate graph statistics independently
# for a small deterministic sample of transactions.
sample = df.sample(n=30, random_state=42)

failures = []

for row in sample.itertuples(index=False):
    previous = df[df["timestamp"] < row.timestamp]

    sender_out = previous.loc[
        previous["nameOrig"] == row.nameOrig,
        "nameDest"
    ].nunique()

    sender_in = previous.loc[
        previous["nameDest"] == row.nameOrig,
        "nameOrig"
    ].nunique()

    receiver_in = previous.loc[
        previous["nameDest"] == row.nameDest,
        "nameOrig"
    ].nunique()

    receiver_out = previous.loc[
        previous["nameOrig"] == row.nameDest,
        "nameDest"
    ].nunique()

    pair_count = int((
        (previous["nameOrig"] == row.nameOrig)
        & (previous["nameDest"] == row.nameDest)
    ).sum())

    expected = [
        sender_out,
        sender_in,
        receiver_in,
        receiver_out,
        pair_count
    ]

    actual = [getattr(row, name) for name in FEATURES]

    if expected != actual:
        failures.append({
            "original_row_id": row.original_row_id,
            "expected": expected,
            "actual": actual
        })

print("Transactions checked:", len(sample))
print("Mismatches:", len(failures))

if failures:
    for item in failures[:5]:
        print(item)
    raise AssertionError(
        "Graph feature validation failed."
    )

print("\nALL SAMPLED GRAPH FEATURES VERIFIED")
print("Graph feature validation passed.")
