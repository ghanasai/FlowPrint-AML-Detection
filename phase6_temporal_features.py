
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# FLOWPRINT - PHASE 6
# Step 6.1: Temporal Transaction Features
# ==========================================

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

print("=" * 55)
print("STEP 6.1 - TEMPORAL TRANSACTION FEATURES")
print("=" * 55)

parts = []

for name in ["train", "validation", "test"]:
    df_part = pd.read_csv(
        data_dir / f"{name}.csv",
        usecols=[
            "transaction_id",
            "timestamp",
            "nameOrig",
            "nameDest"
        ],
        parse_dates=["timestamp"]
    )
    parts.append(df_part)

df = pd.concat(parts, ignore_index=True)

df = df.sort_values(
    ["timestamp", "transaction_id"]
).reset_index(drop=True)

# Compute sender intervals
df["sender_prev_time"] = (
    df.groupby("nameOrig")["timestamp"].shift(1)
)

# Compute receiver intervals
df["receiver_prev_time"] = (
    df.groupby("nameDest")["timestamp"].shift(1)
)

df["sender_gap_minutes"] = (
    (df["timestamp"] - df["sender_prev_time"])
    .dt.total_seconds() / 60
)

df["receiver_gap_minutes"] = (
    (df["timestamp"] - df["receiver_prev_time"])
    .dt.total_seconds() / 60
)

# Prevent equal-timestamp events from appearing as history.
# A zero-minute gap is ambiguous; mark it missing.
df.loc[
    df["sender_gap_minutes"] == 0,
    "sender_gap_minutes"
] = np.nan

df.loc[
    df["receiver_gap_minutes"] == 0,
    "receiver_gap_minutes"
] = np.nan

# Preserve missingness explicitly.
df["sender_has_history"] = (
    df["sender_gap_minutes"].notna().astype(int)
)

df["receiver_has_history"] = (
    df["receiver_gap_minutes"].notna().astype(int)
)

# Use -1 to represent unavailable previous timestamps.
df["sender_gap_minutes"] = (
    df["sender_gap_minutes"].fillna(-1)
)

df["receiver_gap_minutes"] = (
    df["receiver_gap_minutes"].fillna(-1)
)

output_columns = [
    "transaction_id",
    "sender_gap_minutes",
    "receiver_gap_minutes",
    "sender_has_history",
    "receiver_has_history"
]

output_file = data_dir / "temporal_graph_features.csv"

df[output_columns].to_csv(
    output_file,
    index=False
)

print("\nFeature matrix shape:", df[output_columns].shape)

print("\nFirst 5 records:")
print(df[output_columns].head().to_string(index=False))

print("\nSender history available:",
      df["sender_has_history"].sum())

print("Receiver history available:",
      df["receiver_has_history"].sum())

print("\nSaved:", output_file)
print("\nStep 6.1 completed successfully!")
