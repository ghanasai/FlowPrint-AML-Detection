
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

print("STEP 22A: TEMPORAL FOLD AUDIT")
print("=" * 60)

# Load the original AML labels in row order.
labels = pd.read_csv(
    BASE.parent / "AMLNet_v2_transactions.csv",
    usecols=["isMoneyLaundering"]
)

labels["original_row_id"] = np.arange(len(labels))

# Load timestamps already extracted in earlier steps.
timestamps = pd.read_csv(
    DATA / "historical_features.csv",
    usecols=["original_row_id", "timestamp"],
    parse_dates=["timestamp"]
)

assert labels["original_row_id"].is_unique
assert timestamps["original_row_id"].is_unique

df = labels.merge(
    timestamps,
    on="original_row_id",
    how="inner",
    validate="one_to_one"
)

assert len(df) == len(labels)
assert df["timestamp"].notna().all()

df = df.sort_values(
    ["timestamp", "original_row_id"]
).reset_index(drop=True)

print("\nTotal transactions:", len(df))
print("Total AML:", int(df["isMoneyLaundering"].sum()))
print("Earliest timestamp:", df["timestamp"].min())
print("Latest timestamp:", df["timestamp"].max())

# Divide the timeline into ten consecutive
# groups with approximately equal transaction counts.
df["time_decile"] = (
    np.arange(len(df)) * 10 // len(df)
) + 1

summary = (
    df.groupby("time_decile")
    .agg(
        Transactions=("isMoneyLaundering", "size"),
        AML=("isMoneyLaundering", "sum"),
        Start=("timestamp", "min"),
        End=("timestamp", "max")
    )
    .reset_index()
)

summary["Normal"] = (
    summary["Transactions"] - summary["AML"]
)

summary["AML_Rate_Percent"] = (
    100 * summary["AML"] / summary["Transactions"]
)

print("\nCHRONOLOGICAL AML DISTRIBUTION")
print("-" * 60)
print(summary.to_string(index=False))

summary.to_csv(
    RESULTS / "temporal_fold_audit.csv",
    index=False
)

print("\nCUMULATIVE AML DISTRIBUTION")

summary["Cumulative_AML"] = summary["AML"].cumsum()

print(
    summary[
        ["time_decile", "AML", "Cumulative_AML"]
    ].to_string(index=False)
)

print("\nSTEP 22A COMPLETED SUCCESSFULLY")
