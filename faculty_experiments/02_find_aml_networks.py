
from pathlib import Path
import pandas as pd
import networkx as nx

BASE = Path(__file__).resolve().parent.parent
OUTPUT = Path(__file__).resolve().parent / "results"
OUTPUT.mkdir(exist_ok=True)

df = pd.read_csv(
    BASE / "AMLNet_v2_transactions.csv",
    usecols=[
        "nameOrig",
        "nameDest",
        "amount",
        "isMoneyLaundering"
    ]
)

aml = df[df["isMoneyLaundering"] == 1].copy()

# Build AML-only graph to discover connected groups
G = nx.Graph()

G.add_edges_from(
    aml[["nameOrig", "nameDest"]].itertuples(
        index=False, name=None
    )
)

components = sorted(
    nx.connected_components(G),
    key=len,
    reverse=True
)

print("Total AML connected components:", len(components))

print("\nTop 10 connected AML networks:")
for i, nodes in enumerate(components[:10], start=1):
    edges = aml[
        aml["nameOrig"].isin(nodes)
        & aml["nameDest"].isin(nodes)
    ]

    print(
        f"Network {i}: "
        f"{len(nodes)} accounts, "
        f"{len(edges)} AML transactions"
    )

# Save the largest AML connected network
largest_nodes = components[0]

largest_aml = aml[
    aml["nameOrig"].isin(largest_nodes)
    & aml["nameDest"].isin(largest_nodes)
].copy()

largest_aml.to_csv(
    OUTPUT / "largest_aml_network.csv",
    index=False
)

print("\nLargest AML network saved.")
print("\nExample A -> B transactions:")

print(
    largest_aml[
        ["nameOrig", "nameDest", "amount"]
    ].head(10).to_string(index=False)
)
