
import pandas as pd
import networkx as nx
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

print("=" * 55)
print("STEP 7.2 - STRUCTURAL ROLE DISCOVERY")
print("=" * 55)

df = pd.read_csv(
    RESULTS / "validation_suspicious_transactions.csv"
)

flagged = df[df["predicted_suspicious"] == 1].copy()

# Directed graph retains transfer direction
G = nx.MultiDiGraph()

for row in flagged.itertuples(index=False):
    G.add_edge(
        row.nameOrig,
        row.nameDest,
        transaction_id=row.transaction_id,
        amount=row.amount
    )

components = sorted(
    nx.weakly_connected_components(G),
    key=len,
    reverse=True
)

records = []

for network_id, nodes in enumerate(components, start=1):
    subgraph = G.subgraph(nodes)

    for account in subgraph.nodes():
        incoming = subgraph.in_degree(account)
        outgoing = subgraph.out_degree(account)

        unique_senders = len(set(
            subgraph.predecessors(account)
        ))

        unique_receivers = len(set(
            subgraph.successors(account)
        ))

        # Simple, interpretable structural heuristics
        if incoming == 0 and outgoing > 0:
            role = "Source-like"

        elif outgoing == 0 and incoming > 0:
            role = "Destination-like"

        elif unique_receivers >= 2 and incoming >= 1:
            role = "Splitter-like"

        elif unique_senders >= 2 and outgoing >= 1:
            role = "Convergence-like"

        elif incoming > 0 and outgoing > 0:
            role = "Intermediary-like"

        else:
            role = "Unclassified"

        records.append({
            "network_id": network_id,
            "account": account,
            "incoming_transactions": incoming,
            "outgoing_transactions": outgoing,
            "unique_senders": unique_senders,
            "unique_receivers": unique_receivers,
            "structural_role": role
        })

roles_df = pd.DataFrame(records)

roles_df.to_csv(
    RESULTS / "account_structural_roles.csv",
    index=False
)

print("\nTotal accounts analyzed:", len(roles_df))

print("\nSTRUCTURAL ROLE DISTRIBUTION")
print(
    roles_df["structural_role"]
    .value_counts()
    .to_string()
)

print("\nLARGEST NETWORK ACCOUNT ROLES")
print(
    roles_df[roles_df["network_id"] == 1]
    .to_string(index=False)
)

print("\nSaved:",
      RESULTS / "account_structural_roles.csv")

print("\nStep 7.2 completed successfully!")
