
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA = BASE / "data"

print("STEP 15A: HISTORICAL FEATURE AUDIT")
print("=" * 55)

history = pd.read_csv(
    DATA / "historical_features.csv",
    parse_dates=["timestamp"]
)

train = pd.read_csv(DATA / "comparison_train.csv")
val = pd.read_csv(DATA / "comparison_validation.csv")
test = pd.read_csv(DATA / "comparison_test.csv")

FEATURES = [
    "sender_prev_tx_count",
    "receiver_prev_tx_count",
    "sender_prev_total_amount",
    "receiver_prev_total_amount",
    "sender_prev_avg_amount",
    "receiver_prev_avg_amount"
]

print("\n1. DATASET INTEGRITY")
print("Historical rows:", len(history))
print("Unique IDs:", history["original_row_id"].nunique())
print("Missing timestamps:", history["timestamp"].isna().sum())
print("Missing features:", history[FEATURES].isna().sum().sum())

assert len(history) == 1_090_000
assert history["original_row_id"].is_unique
assert not history["timestamp"].isna().any()
assert not history[FEATURES].isna().any().any()

print("\n2. SPLIT INTEGRITY")

sets = [
    set(x["original_row_id"])
    for x in (train, val, test)
]

assert sets[0].isdisjoint(sets[1])
assert sets[0].isdisjoint(sets[2])
assert sets[1].isdisjoint(sets[2])

assert len(set.union(*sets)) == len(history)

for name, frame in zip(
    ["Train", "Validation", "Test"],
    [train, val, test]
):
    print(name, len(frame))

print("\n3. TIMESTAMP TIES")

counts = history["timestamp"].value_counts()
ties = counts[counts > 1]

print("Distinct timestamps:", len(counts))
print("Timestamps with multiple transactions:", len(ties))
print("Transactions involved in ties:", int(ties.sum()))
print("Largest timestamp group:", int(counts.max()))

print("\n4. TIMESTAMP SPLIT BOUNDARIES")

lookup = history.set_index("original_row_id")["timestamp"]

ranges = []

for name, frame in zip(
    ["Train", "Validation", "Test"],
    [train, val, test]
):
    times = lookup.loc[frame["original_row_id"]]
    minimum = times.min()
    maximum = times.max()
    ranges.append((minimum, maximum))
    print(f"{name}: {minimum} to {maximum}")

print("\n5. CHRONOLOGICAL SPLIT CHECK")

train_ok = ranges[0][1] <= ranges[1][0]
val_ok = ranges[1][1] <= ranges[2][0]

print("Train before validation:", train_ok)
print("Validation before test:", val_ok)

if not (train_ok and val_ok):
    print("WARNING: Split boundaries overlap in time.")

print("\n6. SAME-TIMESTAMP ACCOUNT ACTIVITY")

# Check whether simultaneous transactions share accounts.
source = pd.read_csv(
    ROOT / "AMLNet_v2_transactions.csv",
    usecols=["nameOrig", "nameDest"]
)

source["original_row_id"] = np.arange(len(source))

merged = history[
    ["original_row_id", "timestamp"]
].merge(
    source,
    on="original_row_id",
    validate="one_to_one"
)

tied = merged[
    merged["timestamp"].isin(ties.index)
].copy()

if len(tied):
    accounts = pd.concat([
        tied[["timestamp", "nameOrig"]].rename(
            columns={"nameOrig": "account"}
        ),
        tied[["timestamp", "nameDest"]].rename(
            columns={"nameDest": "account"}
        )
    ], ignore_index=True)

    repeated = (
        accounts.groupby(["timestamp", "account"])
        .size()
    )

    repeated = repeated[repeated > 1]

    print("Repeated account/timestamp combinations:",
          len(repeated))
    print("Maximum account appearances at same timestamp:",
          int(repeated.max()) if len(repeated) else 0)
else:
    print("No timestamp ties found.")

print("\nAUDIT COMPLETED")
print("Review timestamp ties before claiming leakage-free results.")
