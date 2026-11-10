
import pandas as pd
from pathlib import Path

# ==========================================
# FLOWPRINT - PHASE 3
# Step 3.1: Feature Selection
# ==========================================

base_dir = Path(__file__).resolve().parent
file_path = base_dir / "AMLNet_v2_transactions.csv"

print("=" * 55)
print("PHASE 3 - FEATURE SELECTION")
print("=" * 55)

# Load selected transaction features
columns = [
    "step",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "nameOrig",
    "nameDest",
    "isMoneyLaundering",
    "laundering_typology"
]

df = pd.read_csv(file_path, usecols=columns)

print("\nDataset shape:", df.shape)

# Ground-truth labels: NOT model inputs
y = df["isMoneyLaundering"].copy()

# Numerical candidate features
feature_columns = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

X = df[feature_columns].copy()

print("\nFeature matrix shape:", X.shape)
print("Target shape:", y.shape)

print("\nSelected model features:")
print(X.columns.tolist())

print("\nAML distribution:")
print(y.value_counts())

print("\nMissing feature values:")
print(X.isnull().sum())

# Inspect the time-step column separately
print("\nStep column statistics:")
print("Minimum:", df["step"].min())
print("Maximum:", df["step"].max())
print("Unique values:", df["step"].nunique())

print("\nFirst 5 feature records:")
print(X.head().to_string(index=False))

print("\nStep 3.1 completed successfully!")




# ==========================================
# Step 3.2: Temporal Data Audit
# ==========================================

print("\n" + "=" * 55)
print("STEP 3.2 - TEMPORAL DATA AUDIT")
print("=" * 55)

# Inspect date and time columns
time_columns = [
    "step",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

print("\nTime column statistics:")
print(df[time_columns].describe().to_string())

print("\nUnique values in time columns:")

for column in time_columns:
    print(
        column,
        "->",
        df[column].nunique()
    )

# Examine month distribution
print("\nTransactions by month:")
print(df["month"].value_counts().sort_index())

# Examine day-of-month distribution
print("\nTransactions by day of month:")
print(df["day_of_month"].value_counts().sort_index())

# Examine hour distribution
print("\nTransactions by hour:")
print(df["hour"].value_counts().sort_index())

# Check whether calendar fields form valid combinations
print("\nUnique calendar combinations:")

calendar_columns = [
    "month",
    "day_of_month",
    "hour"
]

print(
    df[calendar_columns]
    .drop_duplicates()
    .shape[0]
)

# Inspect metadata timestamps without parsing
metadata_preview = pd.read_csv(
    file_path,
    usecols=["metadata"],
    nrows=5
)

print("\nFirst 5 metadata samples:")

for i, value in enumerate(metadata_preview["metadata"]):
    print(f"\nTransaction {i}:")
    print(str(value)[:400])

print("\nStep 3.2 completed successfully!")

