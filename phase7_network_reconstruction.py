
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

print("=" * 55)
print("STEP 7.1 - SUSPICIOUS NETWORK RECONSTRUCTION")
print("=" * 55)

df = pd.read_csv(
    RESULTS / "validation_suspicious_transactions.csv"
)

# Use model predictions, not ground-truth AML labels
flagged = df[df["predicted_suspicious"] == 1].copy()

print("\nFlagged transactions:", len(flagged))

# Build a directed transaction network
G = nx.MultiDiGraph()

for row in flagged.itertuples(index=False):
    G.add_edge(
        row.nameOrig,
        row.nameDest,
        transaction_id=row.transaction_id,
        amount=row.amount,
        score=row.suspiciousness_score
    )

print("Accounts in network:", G.number_of_nodes())
print("Flagged transaction edges:", G.number_of_edges())

# Identify weakly connected candidate networks
components = sorted(
    nx.weakly_connected_components(G),
    key=len,
    reverse=True
)

print("Candidate networks:", len(components))

summary = []

for i, nodes in enumerate(components, start=1):
    subgraph = G.subgraph(nodes)

    summary.append({
        "network_id": i,
        "accounts": len(nodes),
        "transactions": subgraph.number_of_edges(),
        "total_amount": sum(
            d["amount"]
            for _, _, d in subgraph.edges(data=True)
        )
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    RESULTS / "candidate_network_summary.csv",
    index=False
)

print("\nLargest candidate networks:")
print(summary_df.head(10).to_string(index=False))

# Visualize the largest candidate network
if components:
    largest = G.subgraph(components[0]).copy()

    plt.figure(figsize=(12, 8))

    pos = nx.spring_layout(
        largest,
        seed=42
    )

    nx.draw_networkx(
        largest,
        pos,
        with_labels=False,
        node_size=120,
        arrows=True,
        alpha=0.8
    )

    plt.title(
        "FlowPrint - Largest Candidate Suspicious Network"
    )
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        RESULTS / "largest_candidate_network.png",
        dpi=200
    )

    plt.close()

print("\nSaved candidate network summary and visualization.")
print("Step 7.1 completed successfully!")
