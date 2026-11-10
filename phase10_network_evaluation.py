
import pandas as pd
import networkx as nx
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

print("=" * 60)
print("PHASE 10.3 - NETWORK RECONSTRUCTION EVALUATION")
print("=" * 60)

df = pd.read_csv(
    RESULTS / "validation_suspicious_transactions.csv"
)

required = [
    "transaction_id", "nameOrig", "nameDest",
    "isMoneyLaundering", "predicted_suspicious"
]

assert all(col in df.columns for col in required)
assert df["transaction_id"].is_unique

true_aml = df[df["isMoneyLaundering"] == 1].copy()
flagged = df[df["predicted_suspicious"] == 1].copy()

detected_aml = flagged[
    flagged["isMoneyLaundering"] == 1
].copy()

# 1. Account coverage
true_accounts = (
    set(true_aml["nameOrig"])
    | set(true_aml["nameDest"])
)

flagged_accounts = (
    set(flagged["nameOrig"])
    | set(flagged["nameDest"])
)

account_coverage = (
    len(true_accounts & flagged_accounts)
    / len(true_accounts)
    if true_accounts else 0
)

# 2. Transaction and directed-pair recovery
transaction_recall = (
    len(detected_aml) / len(true_aml)
    if len(true_aml) else 0
)

true_pairs = set(zip(
    true_aml["nameOrig"], true_aml["nameDest"]
))

recovered_pairs = set(zip(
    detected_aml["nameOrig"],
    detected_aml["nameDest"]
))

pair_recall = (
    len(recovered_pairs) / len(true_pairs)
    if true_pairs else 0
)

# 3. Ground-truth AML components
G_true = nx.Graph()
G_true.add_edges_from(true_pairs)

true_components = sorted(
    nx.connected_components(G_true),
    key=len,
    reverse=True
)

# Candidate networks from flagged transactions
G_pred = nx.Graph()
G_pred.add_edges_from(zip(
    flagged["nameOrig"], flagged["nameDest"]
))

pred_components = list(
    nx.connected_components(G_pred)
)

# Map every predicted account to its candidate network
account_to_pred = {}

for i, component in enumerate(pred_components):
    for account in component:
        account_to_pred[account] = i

component_results = []

for i, true_component in enumerate(true_components, start=1):

    # Number of true AML accounts found in each candidate
    overlaps = {}

    for account in true_component:
        pred_id = account_to_pred.get(account)

        if pred_id is not None:
            overlaps[pred_id] = (
                overlaps.get(pred_id, 0) + 1
            )

    largest_overlap = max(
        overlaps.values(), default=0
    )

    coverage = (
        largest_overlap / len(true_component)
    )

    component_results.append({
        "true_component_id": i,
        "true_accounts": len(true_component),
        "largest_predicted_overlap": largest_overlap,
        "component_coverage": coverage,
        "fully_recovered": int(
            largest_overlap == len(true_component)
        )
    })

component_df = pd.DataFrame(component_results)

# 4. Candidate network purity
purity_results = []

for i, component in enumerate(pred_components, start=1):
    network_rows = flagged[
        flagged["nameOrig"].isin(component)
        & flagged["nameDest"].isin(component)
    ]

    total = len(network_rows)
    true_positive = int(
        network_rows["isMoneyLaundering"].sum()
    )

    purity_results.append({
        "network_id": i,
        "accounts": len(component),
        "transactions": total,
        "true_aml_transactions": true_positive,
        "purity": true_positive / total if total else 0
    })

purity_df = pd.DataFrame(purity_results)

print("\nACCOUNT COVERAGE")
print("True AML accounts:", len(true_accounts))
print("Recovered AML accounts:",
      len(true_accounts & flagged_accounts))
print("Account coverage:", round(account_coverage, 4))

print("\nEDGE RECOVERY")
print("AML transaction recall:", round(transaction_recall, 4))
print("Directed AML pair recall:", round(pair_recall, 4))

print("\nCOMPONENT RECOVERY")
print("True AML components:", len(true_components))
print("Predicted components:", len(pred_components))
print("Mean largest-component coverage:",
      round(component_df["component_coverage"].mean(), 4))
print("Fully covered true components:",
      int(component_df["fully_recovered"].sum()))

print("\nNETWORK PURITY")
print("Mean network purity:",
      round(purity_df["purity"].mean(), 4))
print("Transaction-weighted purity:",
      round(
          purity_df["true_aml_transactions"].sum()
          / purity_df["transactions"].sum(), 4
      ))

component_df.to_csv(
    RESULTS / "network_component_recovery.csv",
    index=False
)

purity_df.to_csv(
    RESULTS / "network_purity_results.csv",
    index=False
)

print("\nSaved network evaluation results.")
print("Phase 10.3 completed successfully!")
