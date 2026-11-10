
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

SOURCE = ROOT / "AMLNet_v2_transactions.csv"
OUTPUT = DATA / "historical_features.csv"

print("Loading original FlowPrint dataset...")

df = pd.read_csv(
    SOURCE,
    usecols=[
        "metadata",
        "nameOrig",
        "nameDest",
        "amount",
        "isMoneyLaundering"
    ]
)

df["original_row_id"] = np.arange(len(df))

# Extract timestamps without evaluating metadata strings
pattern = re.compile(
    r"""['"]timestamp['"]\s*:\s*datetime\.datetime\(
    \s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,
    \s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)
    (?:\s*,\s*(\d+))?\s*\)""",
    re.VERBOSE
)

def extract_timestamp(value):
    match = pattern.search(str(value))

    if not match:
        return pd.NaT

    parts = [
        int(x) if x is not None else 0
        for x in match.groups()
    ]

    try:
        return pd.Timestamp(*parts)
    except (ValueError, TypeError):
        return pd.NaT

print("Extracting transaction timestamps...")

df["timestamp"] = df["metadata"].map(
    extract_timestamp
)

if df["timestamp"].isna().any():
    raise ValueError(
        "Missing timestamps: "
        f"{df['timestamp'].isna().sum()}"
    )

df = df.drop(columns=["metadata"])

# Sort chronologically, with a deterministic tie-breaker
df = df.sort_values(
    ["timestamp", "original_row_id"]
).reset_index(drop=True)

print("Computing historical account features...")

# Previous outgoing transaction count
df["sender_prev_tx_count"] = (
    df.groupby("nameOrig").cumcount()
)

# Previous incoming transaction count
df["receiver_prev_tx_count"] = (
    df.groupby("nameDest").cumcount()
)

# Previous outgoing amount totals
sender_cumulative = (
    df.groupby("nameOrig")["amount"].cumsum()
)

df["sender_prev_total_amount"] = (
    sender_cumulative - df["amount"]
).clip(lower=0)

# Previous incoming amount totals
receiver_cumulative = (
    df.groupby("nameDest")["amount"].cumsum()
)

df["receiver_prev_total_amount"] = (
    receiver_cumulative - df["amount"]
).clip(lower=0)

# Historical averages
df["sender_prev_avg_amount"] = (
    df["sender_prev_total_amount"]
    / df["sender_prev_tx_count"].replace(0, np.nan)
).fillna(0)

df["receiver_prev_avg_amount"] = (
    df["receiver_prev_total_amount"]
    / df["receiver_prev_tx_count"].replace(0, np.nan)
).fillna(0)

FEATURES = [
    "sender_prev_tx_count",
    "receiver_prev_tx_count",
    "sender_prev_total_amount",
    "receiver_prev_total_amount",
    "sender_prev_avg_amount",
    "receiver_prev_avg_amount"
]

# Verify historical features are valid
assert not df[FEATURES].isna().any().any()
assert (df[FEATURES] >= 0).all().all()

# Verify first observed sender/receiver histories
assert (
    df.loc[df["sender_prev_tx_count"] == 0,
           "sender_prev_total_amount"] == 0
).all()

assert (
    df.loc[df["receiver_prev_tx_count"] == 0,
           "receiver_prev_total_amount"] == 0
).all()

# Keep a compact feature file keyed by original row ID
output_columns = [
    "original_row_id",
    "timestamp",
    "isMoneyLaundering"
] + FEATURES

df[output_columns].to_csv(
    OUTPUT,
    index=False
)

print("\nHISTORICAL FEATURE ENGINEERING COMPLETE")
print("---------------------------------------")
print("Transactions processed:", len(df))
print("Unique senders:", df["nameOrig"].nunique())
print("Unique receivers:", df["nameDest"].nunique())
print("Historical features created:", len(FEATURES))
print("Missing feature values:",
      int(df[FEATURES].isna().sum().sum()))

print("\nSample historical features:")
print(
    df[[
        "original_row_id",
        "sender_prev_tx_count",
        "receiver_prev_tx_count",
        "sender_prev_total_amount",
        "receiver_prev_total_amount"
    ]].head(10).to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT)
