
from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(
    ROOT / "AMLNet_v2_transactions.csv",
    usecols=[
        "nameOrig", "nameDest",
        "amount", "isMoneyLaundering"
    ]
)

# Real example from the dataset
A = "C3108"
B = "C403"

# Find direct neighbors of A and B
direct = df[
    df["nameOrig"].isin([A, B])
    | df["nameDest"].isin([A, B])
].copy()

# Choose a readable subgraph while preserving
# the actual A-B transactions.
seed = direct[
    (direct["nameOrig"] == A)
    & (direct["nameDest"] == B)
]

# Prioritize AML links, then other connected links
other = direct.drop(index=seed.index)
other = other.sort_values(
    ["isMoneyLaundering", "amount"],
    ascending=[False, False]
)

selected = pd.concat(
    [seed, other.head(35)]
).drop_duplicates()

# Aggregate repeated account pairs for display
edges = (
    selected.groupby(
        ["nameOrig", "nameDest"], as_index=False
    )
    .agg(
        amount=("amount", "sum"),
        count=("amount", "size"),
        aml=("isMoneyLaundering", "max")
    )
)

G = nx.DiGraph()

for row in edges.itertuples(index=False):
    G.add_edge(
        row.nameOrig,
        row.nameDest,
        amount=row.amount,
        count=row.count,
        aml=row.aml
    )

pos = nx.spring_layout(G, seed=42, k=1.2)

plt.figure(figsize=(16, 11))

node_colors = [
    "#0d9488" if n in [A, B] else "#94a3b8"
    for n in G.nodes()
]

edge_colors = [
    "#dc2626" if G[u][v]["aml"] else "#64748b"
    for u, v in G.edges()
]

nx.draw_networkx_nodes(
    G, pos,
    node_color=node_colors,
    node_size=1050,
    edgecolors="white",
    linewidths=1.5
)

nx.draw_networkx_labels(
    G, pos,
    font_size=8,
    font_weight="bold"
)

nx.draw_networkx_edges(
    G, pos,
    edge_color=edge_colors,
    arrows=True,
    arrowsize=18,
    width=1.8,
    connectionstyle="arc3,rad=0.08"
)

# Label the main A -> B relationship
labels = {}

if G.has_edge(A, B):
    e = G[A][B]
    labels[(A, B)] = (
        f"{e['count']} txn\n"
        f"{e['amount']:,.0f}"
    )

nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels=labels,
    font_size=8,
    rotate=False
)

legend = [
    Line2D([0], [0], color="#dc2626", lw=3,
           label="AML-labeled link"),
    Line2D([0], [0], color="#64748b", lw=3,
           label="Normal-only link"),
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor="#0d9488",
           markersize=12, label="Selected accounts A and B")
]

plt.legend(handles=legend, loc="upper left")

plt.title(
    f"Real Account Relationship Network: {A} → {B}\n"
    "AMLNet v2 | Direct Account Connections",
    fontsize=17,
    fontweight="bold"
)

plt.axis("off")
plt.tight_layout()

image_path = OUT / "real_account_relationship.png"
plt.savefig(image_path, dpi=300, bbox_inches="tight")

edges.to_csv(
    OUT / "real_account_relationship_edges.csv",
    index=False
)

print("Selected accounts:", A, "->", B)
print("Accounts displayed:", G.number_of_nodes())
print("Unique directed links:", G.number_of_edges())
print("Transactions selected:", len(selected))
print("Graph saved:", image_path)

plt.show()
