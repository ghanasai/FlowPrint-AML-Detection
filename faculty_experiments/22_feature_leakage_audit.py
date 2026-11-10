
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA = BASE / "data"
RESULTS = BASE / "results"

RESULTS.mkdir(parents=True, exist_ok=True)

print("STEP 20A: FEATURE LEAKAGE AND AVAILABILITY AUDIT")
print("=" * 65)

ORIGINAL = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

HISTORICAL = [
    "sender_prev_tx_count",
    "receiver_prev_tx_count",
    "sender_prev_total_amount",
    "receiver_prev_total_amount",
    "sender_prev_avg_amount",
    "receiver_prev_avg_amount"
]

GRAPH = [
    "sender_out_degree",
    "sender_in_degree",
    "receiver_in_degree",
    "receiver_out_degree",
    "pair_previous_count"
]

ALL_FEATURES = ORIGINAL + HISTORICAL + GRAPH

print("\n1. Loading dataset...")

df = pd.read_csv(
    ROOT / "AMLNet_v2_transactions.csv",
    usecols=[
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "isMoneyLaundering"
    ]
)

print("Transactions:", len(df))
print("AML transactions:", int(df["isMoneyLaundering"].sum()))

print("\n2. Checking balance relationships...")

expected_balance = df["oldbalanceOrg"] - df["amount"]

difference = (
    df["newbalanceOrig"] - expected_balance
).abs()

for tolerance in [0.01, 1.0, 10.0]:
    matches = (difference <= tolerance).mean()

    print(
        f"newbalanceOrig approximately equals "
        f"oldbalanceOrg - amount "
        f"(tolerance={tolerance}): "
        f"{matches:.2%}"
    )

print("\n3. Checking label associations...")

for label, group in df.groupby("isMoneyLaundering"):
    print(f"\nClass {label}:")
    print("Transactions:", len(group))
    print("Mean amount:", group["amount"].mean())
    print("Mean old balance:", group["oldbalanceOrg"].mean())
    print("Mean new balance:", group["newbalanceOrig"].mean())

print("\n4. Checking generated feature files...")

history = pd.read_csv(
    DATA / "historical_features_corrected.csv",
    usecols=["original_row_id"] + HISTORICAL
)

graph = pd.read_csv(
    DATA / "graph_features.csv",
    usecols=["original_row_id"] + GRAPH
)

timestamps = pd.read_csv(
    DATA / "historical_features.csv",
    usecols=["original_row_id", "timestamp"],
    parse_dates=["timestamp"]
)

assert len(history) == len(df)
assert len(graph) == len(df)
assert len(timestamps) == len(df)

for name, frame in [
    ("history", history),
    ("graph", graph),
    ("timestamps", timestamps)
]:
    assert frame["original_row_id"].is_unique, (
        f"Duplicate transaction IDs in {name}"
    )

assert not history[HISTORICAL].isna().any().any()
assert not graph[GRAPH].isna().any().any()
assert timestamps["timestamp"].notna().all()

print("Historical features: PASS")
print("Graph features: PASS")
print("Timestamp completeness: PASS")

print("\n5. Feature availability classification...")

rows = []

for feature in ALL_FEATURES:

    if feature == "newbalanceOrig":
        status = "CONDITIONAL"
        reason = (
            "Post-transaction balance; availability "
            "depends on prediction timing."
        )

    elif feature == "oldbalanceOrg":
        status = "VERIFY"
        reason = (
            "Confirm balance is recorded before "
            "the transaction."
        )

    elif feature in HISTORICAL:
        status = "VERIFY"
        reason = (
            "Confirm strictly earlier transactions "
            "and correct event-time ordering."
        )

    elif feature in GRAPH:
        status = "VERIFY"
        reason = (
            "Confirm graph uses strictly earlier "
            "connections and transfers."
        )

    else:
        status = "CONDITIONAL"
        reason = (
            "Confirm feature is known when "
            "the AML decision is made."
        )

    rows.append({
        "Feature": feature,
        "Status": status,
        "Reason": reason
    })

audit = pd.DataFrame(rows)

print(audit.to_string(index=False))

audit.to_csv(
    RESULTS / "feature_availability_audit.csv",
    index=False
)

print("\n6. Proposed deployment feature set...")

DEPLOYMENT_FEATURES = [
    feature for feature in ALL_FEATURES
    if feature != "newbalanceOrig"
]

print("Original features:", len(ALL_FEATURES))
print("Proposed features:", len(DEPLOYMENT_FEATURES))
print("Excluded:", ["newbalanceOrig"])

print("\nIMPORTANT:")
print(
    "This audit checks data integrity and flags "
    "features requiring availability verification."
)
print(
    "It does not prove that the remaining "
    "features are leakage-free."
)

print("\nSTEP 20A COMPLETED")
