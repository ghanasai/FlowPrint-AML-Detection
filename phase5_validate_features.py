
import pandas as pd
from pathlib import Path

# ==========================================
# FLOWPRINT - PHASE 5
# Step 5.2: Historical Feature Validation
# ==========================================

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

print("=" * 55)
print("STEP 5.2 - HISTORICAL FEATURE VALIDATION")
print("=" * 55)

features = pd.read_csv(
    data_dir / "historical_graph_features.csv"
)

print("\nTotal records:", len(features))

# Check transaction ID uniqueness
print(
    "Unique transaction IDs:",
    features["transaction_id"].nunique()
)

print(
    "Duplicate transaction IDs:",
    features["transaction_id"].duplicated().sum()
)

# Check negative feature values
feature_columns = [
    "sender_prev_out_count",
    "receiver_prev_in_count",
    "sender_prev_out_amount",
    "receiver_prev_in_amount"
]

print("\nNegative feature values:")

for column in feature_columns:
    print(column, ":", (features[column] < 0).sum())

# Check how many records have historical activity
print("\nNONZERO HISTORICAL ACTIVITY")

for column in feature_columns:
    nonzero = (features[column] > 0).sum()

    print(
        f"{column}: {nonzero:,} "
        f"({nonzero / len(features) * 100:.2f}%)"
    )

# Summary statistics
print("\nFEATURE STATISTICS")

print(
    features[feature_columns]
    .describe()
    .to_string()
)

# Check later transaction records
print("\nLAST 5 CHRONOLOGICAL RECORDS")

print(
    features.tail(5).to_string(index=False)
)

print("\nStep 5.2 completed successfully!")
