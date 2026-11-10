
import pandas as pd
import networkx as nx
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

print("=" * 55)
print("STEP 7.3 - IMPROVED STRUCTURAL ROLES")
print("=" * 55)

df = pd.read_csv(
    RESULTS / "validation_suspicious_transactions.csv"
)

flagged = df[df["predicted_suspicious"] == 1]

G = nx.MultiDiGraph()

for row in flagged.itertuples(index=False):
    G.add_edge(
        row.nameOrig,
        row.nameDest,
        amount=row.amount,
        transaction_id=row.transaction_id
    )

components = sorted(
    nx.weakly_connected_components(G),
    key=len,
    reverse=True
)

records = []

for network_id, nodes in enumerate(components, 1):
    subgraph = G.subgraph(nodes)

    for account in subgraph.nodes():
        incoming = subgraph.in_degree(account)
        outgoing = subgraph.out_degree(account)

        unique_senders = len(
            set(subgraph.predecessors(account))
        )

        unique_receivers = len(
            set(subgraph.successors(account))
        )

        # Independent structural role indicators
        source = incoming == 0 and outgoing > 0
        destination = outgoing == 0 and incoming > 0

        splitter = unique_receivers >= 2
        convergence = unique_senders >= 2

        intermediary = incoming > 0 and outgoing > 0

        roles = []

        if source:
            roles.append("Source-like")

        if destination:
            roles.append("Destination-like")

        if splitter:
            roles.append("Splitter-like")

        if convergence:
            roles.append("Convergence-like")

        if intermediary:
            roles.append("Intermediary-like")

        records.append({
            "network_id": network_id,
            "account": account,
            "incoming_transactions": incoming,
            "outgoing_transactions": outgoing,
            "unique_senders": unique_senders,
            "unique_receivers": unique_receivers,
            "source_like": int(source),
            "destination_like": int(destination),
            "splitter_like": int(splitter),
            "convergence_like": int(convergence),
            "intermediary_like": int(intermediary),
            "roles": ", ".join(roles)
        })

roles_df = pd.DataFrame(records)

output = RESULTS / "improved_account_roles.csv"
roles_df.to_csv(output, index=False)

print("\nTotal accounts:", len(roles_df))

print("\nROLE COUNTS (OVERLAPPING)")
for column in [
    "source_like",
    "destination_like",
    "splitter_like",
    "convergence_like",
    "intermediary_like"
]:
    print(column, ":", roles_df[column].sum())

print("\nLARGEST NETWORK ROLES")
print(
    roles_df[roles_df["network_id"] == 1][
        ["account", "unique_senders",
         "unique_receivers", "roles"]
    ].to_string(index=False)
)

print("\nSaved:", output)
print("\nStep 7.3 completed successfully!")
