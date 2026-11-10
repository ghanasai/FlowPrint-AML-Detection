
import pandas as pd
import networkx as nx
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

print("=" * 60)
print("PHASE 10.4 - FINAL TEST NETWORK EVALUATION")
print("=" * 60)

# Use frozen test predictions
df = pd.read_csv(
    RESULTS / "final_test_predictions.csv"
)

required = [
    "transaction_id", "nameOrig", "nameDest",
    "isMoneyLaundering", "predicted_suspicious"
]

assert all(c in df.columns for c in required)
assert df["transaction_id"].is_unique
assert df["isMoneyLaundering"].isin([0, 1]).all()
assert df["predicted_suspicious"].isin([0, 1]).all()

true_aml = df[df["isMoneyLaundering"] == 1]
flagged = df[df["predicted_suspicious"] == 1]
detected = flagged[flagged["isMoneyLaundering"] == 1]

# 1. Account coverage
true_accounts = (
    set(true_aml["nameOrig"])
    | set(true_aml["nameDest"])
)

flagged_accounts = (
    set(flagged["nameOrig"])
    | set(flagged["nameDest"])
)

covered_accounts = true_accounts & flagged_accounts

account_coverage = (
    len(covered_accounts) / len(true_accounts)
    if true_accounts else 0
)

# 2. Transaction and directed-pair recovery
transaction_recall = (
    len(detected) / len(true_aml)
    if len(true_aml) else 0
)

true_pairs = set(zip(
    true_aml["nameOrig"], true_aml["nameDest"]
))

detected_pairs = set(zip(
    detected["nameOrig"], detected["nameDest"]
))

pair_recall = (
    len(detected_pairs) / len(true_pairs)
    if true_pairs else 0
)

# 3. Build undirected connectivity graphs
# Direction is preserved for pair recall above.
G_true = nx.Graph()
G_true.add_edges_from(true_pairs)

G_pred = nx.Graph()
G_pred.add_edges_from(zip(
    flagged["nameOrig"], flagged["nameDest"]
))

true_components = sorted(
    nx.connected_components(G_true),
    key=len,
    reverse=True
)

pred_components = sorted(
    nx.connected_components(G_pred),
    key=len,
    reverse=True
)

account_to_pred = {
    account: i
    for i, component in enumerate(pred_components)
    for account in component
}

component_rows = []

for i, component in enumerate(true_components, start=1):
    overlap_counts = {}

    for account in component:
        pred_id = account_to_pred.get(account)

        if pred_id is not None:
            overlap_counts[pred_id] = (
                overlap_counts.get(pred_id, 0) + 1
            )

    largest_overlap = max(
        overlap_counts.values(), default=0
    )

    component_rows.append({
        "true_component_id": i,
        "true_accounts": len(component),
        "largest_predicted_overlap": largest_overlap,
        "component_coverage": (
            largest_overlap / len(component)
        ),
        "fully_covered": int(
            largest_overlap == len(component)
        )
    })

component_df = pd.DataFrame(component_rows)

# 4. Candidate network purity
purity_rows = []

for i, component in enumerate(pred_components, start=1):
    network = flagged[
        flagged["nameOrig"].isin(component)
        & flagged["nameDest"].isin(component)
    ]

    total = len(network)
    aml_count = int(network["isMoneyLaundering"].sum())

    purity_rows.append({
        "network_id": i,
        "accounts": len(component),
        "transactions": total,
        "true_aml_transactions": aml_count,
        "purity": aml_count / total if total else 0
    })

purity_df = pd.DataFrame(purity_rows)

weighted_purity = (
    len(detected) / len(flagged)
    if len(flagged) else 0
)

print("\nACCOUNT COVERAGE")
print("True AML accounts:", len(true_accounts))
print("Recovered AML accounts:", len(covered_accounts))
print("Account coverage:", round(account_coverage, 4))

print("\nEDGE RECOVERY")
print("AML transaction recall:", round(transaction_recall, 4))
print("Directed AML pair recall:", round(pair_recall, 4))

print("\nCOMPONENT RECOVERY")
print("True AML components:", len(true_components))
print("Predicted components:", len(pred_components))
print(
    "Mean largest-component coverage:",
    round(component_df["component_coverage"].mean(), 4)
)
print(
    "Fully covered true components:",
    int(component_df["fully_covered"].sum())
)

print("\nNETWORK PURITY")
print(
    "Mean network purity:",
    round(purity_df["purity"].mean(), 4)
)
print(
    "Transaction-weighted purity:",
    round(weighted_purity, 4)
)

component_df.to_csv(
    RESULTS / "final_test_component_recovery.csv",
    index=False
)

purity_df.to_csv(
    RESULTS / "final_test_network_purity.csv",
    index=False
)

print("\nSaved final test network evaluation files.")
print("Phase 10.4 completed successfully!")
