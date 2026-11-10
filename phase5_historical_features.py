
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ==========================================
# FLOWPRINT - PHASE 5
# Step 5.1: Historical Account Features
# ==========================================

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

print("=" * 55)
print("PHASE 5 - HISTORICAL GRAPH FEATURES")
print("=" * 55)

# Load all chronological partitions
parts = []

for name in ["train", "validation", "test"]:
    part = pd.read_csv(
        data_dir / f"{name}.csv",
        usecols=[
            "transaction_id",
            "timestamp",
            "nameOrig",
            "nameDest",
            "amount"
        ],
        parse_dates=["timestamp"]
    )

    part["split"] = name
    parts.append(part)

df = pd.concat(parts, ignore_index=True)

df = df.sort_values(
    ["timestamp", "transaction_id"]
).reset_index(drop=True)

# Maintain historical account statistics
out_count = defaultdict(int)
in_count = defaultdict(int)

out_amount = defaultdict(float)
in_amount = defaultdict(float)

records = []

print("\nComputing historical features...")

# Process equal-timestamp transactions as a batch
# so they cannot see each other as historical events.
for timestamp, group in df.groupby(
    "timestamp", sort=False
):
    # First: calculate features from previous timestamps
    for row in group.itertuples(index=False):
        sender = row.nameOrig
        receiver = row.nameDest

        records.append({
            "transaction_id": row.transaction_id,
            "sender_prev_out_count": out_count[sender],
            "receiver_prev_in_count": in_count[receiver],
            "sender_prev_out_amount": out_amount[sender],
            "receiver_prev_in_amount": in_amount[receiver]
        })

    # Second: update history after feature extraction
    for row in group.itertuples(index=False):
        sender = row.nameOrig
        receiver = row.nameDest
        amount = row.amount

        out_count[sender] += 1
        in_count[receiver] += 1

        out_amount[sender] += amount
        in_amount[receiver] += amount

features_df = pd.DataFrame(records)

# Save features with stable transaction IDs
output_file = data_dir / "historical_graph_features.csv"

features_df.to_csv(output_file, index=False)

print("\nFeature matrix shape:", features_df.shape)

print("\nFirst 5 records:")
print(features_df.head().to_string(index=False))

print("\nMissing values:")
print(features_df.isna().sum())

print("\nSaved:", output_file)
print("\nStep 5.1 completed successfully!")
