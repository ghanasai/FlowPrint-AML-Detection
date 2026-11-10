
import pandas as pd
from pathlib import Path

# ==========================================
# FLOWPRINT - PHASE 3
# Step 3.4: Chronological Dataset Splitting
# ==========================================

base_dir = Path(__file__).resolve().parent

dataset_file = base_dir / "AMLNet_v2_transactions.csv"
timestamp_file = base_dir / "flowprint_timestamps.csv"

print("=" * 55)
print("STEP 3.4 - CHRONOLOGICAL DATA SPLITTING")
print("=" * 55)

# Load transaction data and timestamps
df = pd.read_csv(
    dataset_file,
    usecols=[
        "nameOrig",
        "nameDest",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "hour",
        "day_of_week",
        "day_of_month",
        "month",
        "isMoneyLaundering",
        "laundering_typology"
    ]
)

timestamps = pd.read_csv(
    timestamp_file,
    parse_dates=["timestamp"]
)

# Verify transaction alignment
assert len(df) == len(timestamps)
assert (
    timestamps["transaction_id"].to_numpy()
    == range(len(df))
).all()

df["transaction_id"] = timestamps["transaction_id"]
df["timestamp"] = timestamps["timestamp"]

# Sort chronologically
df = df.sort_values(
    ["timestamp", "transaction_id"]
).reset_index(drop=True)

print("\nTransactions sorted chronologically.")

# Select split boundaries using unique timestamps
unique_times = df["timestamp"].drop_duplicates().sort_values()

train_boundary = unique_times.iloc[int(len(unique_times) * 0.70)]
val_boundary = unique_times.iloc[int(len(unique_times) * 0.85)]

train_df = df[df["timestamp"] < train_boundary].copy()

val_df = df[
    (df["timestamp"] >= train_boundary)
    & (df["timestamp"] < val_boundary)
].copy()

test_df = df[df["timestamp"] >= val_boundary].copy()

# Save datasets
output_dir = base_dir / "data"
output_dir.mkdir(exist_ok=True)

train_df.to_csv(output_dir / "train.csv", index=False)
val_df.to_csv(output_dir / "validation.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

# Display results
for name, data in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:
    print(f"\n{name}")
    print("Transactions:", len(data))
    print("AML transactions:", data["isMoneyLaundering"].sum())
    print("Normal transactions:",
          (data["isMoneyLaundering"] == 0).sum())
    print("Earliest:", data["timestamp"].min())
    print("Latest:", data["timestamp"].max())

print("\nChronological split completed successfully!")
