
import pandas as pd
from pathlib import Path

# ==========================================
# FLOWPRINT - PHASE 2
# Step 2.1: Identify Unique Graph Nodes
# ==========================================

file_path = Path(__file__).resolve().parent / "AMLNet_v2_transactions.csv"

df = pd.read_csv(
    file_path,
    usecols=[
        "nameOrig",
        "nameDest",
        "amount",
        "type",
        "step",
        "isMoneyLaundering",
        "laundering_typology",
        "hour",
        "day_of_week"
    ]
)

print("=" * 55)
print("PHASE 2 - GRAPH NODE IDENTIFICATION")
print("=" * 55)

# Combine sender and receiver accounts
all_accounts = pd.concat(
    [df["nameOrig"], df["nameDest"]],
    ignore_index=True
)

# Identify unique accounts
unique_accounts = all_accounts.unique()

# Count customer and merchant accounts
customer_nodes = sum(
    str(account).startswith("C")
    for account in unique_accounts
)

merchant_nodes = sum(
    str(account).startswith("M")
    for account in unique_accounts
)

print("\nTotal transactions:", len(df))
print("Total unique accounts:", len(unique_accounts))
print("Customer accounts:", customer_nodes)
print("Merchant accounts:", merchant_nodes)

print("\nFirst 10 account IDs:")
print(unique_accounts[:10])



# ==========================================
# Step 2.2: Numerical Node Mapping
# ==========================================

print("\n" + "=" * 55)
print("STEP 2.2 - NUMERICAL NODE MAPPING")
print("=" * 55)

# Assign a unique integer to every account
node_to_id = {
    account: node_id
    for node_id, account in enumerate(unique_accounts)
}

# Create the reverse mapping
id_to_node = {
    node_id: account
    for account, node_id in node_to_id.items()
}

# Convert account names to numerical IDs
df["src_id"] = df["nameOrig"].map(node_to_id)
df["dst_id"] = df["nameDest"].map(node_to_id)

# Check for mapping errors
print("\nMissing source IDs:", df["src_id"].isna().sum())
print("Missing destination IDs:", df["dst_id"].isna().sum())

# Convert IDs to integers
df["src_id"] = df["src_id"].astype("int32")
df["dst_id"] = df["dst_id"].astype("int32")

print("\nTotal node mappings:", len(node_to_id))

print("\nFirst 5 mapped transactions:")
print(
    df[
        ["nameOrig", "src_id", "nameDest", "dst_id"]
    ].head().to_string(index=False)
)



# ==========================================
# Step 2.3: Directed Graph Edge Construction
# ==========================================

import numpy as np

print("\n" + "=" * 55)
print("STEP 2.3 - DIRECTED GRAPH EDGES")
print("=" * 55)

# Create edge index:
# First row = source IDs
# Second row = destination IDs
edge_index = df[["src_id", "dst_id"]].to_numpy(
    dtype=np.int32
).T

print("\nEdge index shape:", edge_index.shape)

print("\nFirst 5 directed edges:")
for i in range(5):
    print(
        f"Edge {i}: "
        f"{edge_index[0, i]} -> {edge_index[1, i]}"
    )

# Total number of transaction edges
total_edges = edge_index.shape[1]
print("\nTotal transaction edges:", total_edges)

# Count transactions where sender equals receiver
self_loops = (df["src_id"] == df["dst_id"]).sum()
print("Self-loop transactions:", self_loops)

# Count unique directed account pairs
unique_pairs = df[
    ["src_id", "dst_id"]
].drop_duplicates()

print("Unique directed account pairs:", len(unique_pairs))

# Repeated transactions between account pairs
repeated_edges = total_edges - len(unique_pairs)
print("Repeated transaction edges:", repeated_edges)



# ==========================================
# Step 2.4: Account-Level Graph Statistics
# ==========================================

print("\n" + "=" * 55)
print("STEP 2.4 - ACCOUNT-LEVEL GRAPH STATISTICS")
print("=" * 55)

# Create a table containing every account
node_table = pd.DataFrame({
    "node_id": range(len(unique_accounts)),
    "account": unique_accounts
})

# Count outgoing transactions
out_counts = df.groupby("src_id").size()

node_table["out_transactions"] = (
    node_table["node_id"]
    .map(out_counts)
    .fillna(0)
    .astype(int)
)

# Count incoming transactions
in_counts = df.groupby("dst_id").size()

node_table["in_transactions"] = (
    node_table["node_id"]
    .map(in_counts)
    .fillna(0)
    .astype(int)
)

# Count unique receivers per account
unique_receivers = (
    df.groupby("src_id")["dst_id"].nunique()
)

node_table["unique_receivers"] = (
    node_table["node_id"]
    .map(unique_receivers)
    .fillna(0)
    .astype(int)
)

# Count unique senders per account
unique_senders = (
    df.groupby("dst_id")["src_id"].nunique()
)

node_table["unique_senders"] = (
    node_table["node_id"]
    .map(unique_senders)
    .fillna(0)
    .astype(int)
)

# Account type
node_table["node_type"] = (
    node_table["account"]
    .str.startswith("C")
    .map({True: "Customer", False: "Merchant"})
)

# Display results
print("\nNode table shape:", node_table.shape)

print("\nFirst 10 accounts:")
print(node_table.head(10).to_string(index=False))

print("\nTop 10 accounts by outgoing transactions:")
print(
    node_table.nlargest(10, "out_transactions")
    .to_string(index=False)
)

print("\nTop 10 accounts by incoming transactions:")
print(
    node_table.nlargest(10, "in_transactions")
    .to_string(index=False)
)

# Validate transaction counts
print("\nTotal outgoing transactions:",
      node_table["out_transactions"].sum())

print("Total incoming transactions:",
      node_table["in_transactions"].sum())




# ==========================================
# Step 2.5: Transaction Edge Features
# ==========================================

print("\n" + "=" * 55)
print("STEP 2.5 - TRANSACTION EDGE FEATURES")
print("=" * 55)

# Create transaction-level edge table
edge_table = df[
    [
        "src_id",
        "dst_id",
        "amount",
        "type",
        "step",
        "hour",
        "day_of_week",
        "isMoneyLaundering",
        "laundering_typology"
    ]
].copy()

print("\nEdge table shape:")
print(edge_table.shape)

print("\nFirst 5 transaction edges:")
print(edge_table.head().to_string(index=False))

# Count normal and AML transactions
print("\nAML label distribution:")
print(edge_table["isMoneyLaundering"].value_counts())

# Count laundering patterns
print("\nLaundering typologies:")
print(edge_table["laundering_typology"].value_counts())

# Check for missing values
print("\nMissing values:")
print(edge_table.isnull().sum())

# Save the edge table
output_file = "flowprint_edge_table.csv"

edge_table.to_csv(output_file, index=False)

print("\nSaved:", output_file)



# ==========================================
# Step 2.6: AML Transaction Subgraph Analysis
# ==========================================

import networkx as nx

print("\n" + "=" * 55)
print("STEP 2.6 - AML SUBGRAPH ANALYSIS")
print("=" * 55)

# Extract AML-labeled transactions
aml_edges = edge_table[
    edge_table["isMoneyLaundering"] == 1
].copy()

print("\nAML transactions:", len(aml_edges))

# Construct a directed AML-only graph
aml_graph = nx.from_pandas_edgelist(
    aml_edges,
    source="src_id",
    target="dst_id",
    create_using=nx.DiGraph()
)

print("AML graph nodes:", aml_graph.number_of_nodes())
print("Unique directed AML links:", aml_graph.number_of_edges())

# Analyze weakly connected components
components = list(
    nx.weakly_connected_components(aml_graph)
)

component_sizes = sorted(
    [len(component) for component in components],
    reverse=True
)

print("\nNumber of connected components:", len(components))
print("Largest 10 component sizes:", component_sizes[:10])

print(
    "Largest component size:",
    component_sizes[0] if component_sizes else 0
)

# Count components with more than two accounts
multi_account_components = sum(
    size > 2 for size in component_sizes
)

print(
    "Components with more than 2 accounts:",
    multi_account_components
)

# Count AML transaction types
print("\nAML typology distribution:")
print(aml_edges["laundering_typology"].value_counts())

print("\nStep 2.6 completed successfully!")



# ==========================================
# Step 2.7: AML Component Typology Analysis
# ==========================================

print("\n" + "=" * 55)
print("STEP 2.7 - COMPONENT TYPOLOGY ANALYSIS")
print("=" * 55)

# Assign each AML account to a connected component
node_component = {}

for component_id, component in enumerate(components):
    for node in component:
        node_component[node] = component_id

# Attach component IDs to AML transactions
aml_edges["component_id"] = (
    aml_edges["src_id"].map(node_component)
)

# Count transactions in each component
component_transactions = (
    aml_edges.groupby("component_id")
    .size()
    .rename("transaction_count")
)

# Count distinct typologies in each component
component_typologies = (
    aml_edges.groupby("component_id")["laundering_typology"]
    .nunique()
    .rename("typology_count")
)

# Build component summary
component_summary = pd.DataFrame({
    "component_id": range(len(components)),
    "account_count": [len(c) for c in components]
})

component_summary = component_summary.merge(
    component_transactions,
    on="component_id",
    how="left"
).merge(
    component_typologies,
    on="component_id",
    how="left"
)

# Count mixed-typology components
mixed_components = component_summary[
    component_summary["typology_count"] > 1
]

print("\nTotal components:", len(component_summary))
print("Mixed-typology components:", len(mixed_components))

print("\nTypology counts per component:")
print(
    component_summary["typology_count"]
    .value_counts()
    .sort_index()
)

print("\nTop 10 largest components:")
print(
    component_summary
    .sort_values("account_count", ascending=False)
    .head(10)
    .to_string(index=False)
)

# Display typology composition of the largest component
largest_component_id = (
    component_summary
    .sort_values("account_count", ascending=False)
    .iloc[0]["component_id"]
)

largest_component_edges = aml_edges[
    aml_edges["component_id"] == largest_component_id
]

print("\nLargest component typologies:")
print(
    largest_component_edges["laundering_typology"]
    .value_counts()
)

# Save summary
component_summary.to_csv(
    "flowprint_component_summary.csv",
    index=False
)

print("\nSaved: flowprint_component_summary.csv")
print("Step 2.7 completed successfully!")



# ==========================================
# Step 2.8: Visualize Largest AML Component
# ==========================================

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

print("\n" + "=" * 55)
print("STEP 2.8 - AML NETWORK VISUALIZATION")
print("=" * 55)

# Select the largest AML connected component
largest_nodes = max(components, key=len)

# Extract its AML transactions
visual_edges = aml_edges[
    aml_edges["src_id"].isin(largest_nodes)
    & aml_edges["dst_id"].isin(largest_nodes)
].copy()

# Create directed graph
visual_graph = nx.DiGraph()

for row in visual_edges.itertuples(index=False):
    visual_graph.add_edge(row.src_id, row.dst_id)

# Define colors for laundering typologies
typology_colors = {
    "layering": "red",
    "structuring": "orange",
    "integration": "purple"
}

# Use the most frequent typology for each directed pair
pair_typology = (
    visual_edges
    .groupby(["src_id", "dst_id"])["laundering_typology"]
    .agg(lambda x: x.value_counts().idxmax())
)

edge_colors = [
    typology_colors.get(pair_typology.loc[u, v], "gray")
    for u, v in visual_graph.edges()
]

# Create a reproducible graph layout
positions = nx.spring_layout(
    visual_graph,
    seed=42,
    k=0.65,
    iterations=100
)

# Draw graph
plt.figure(figsize=(14, 10))

nx.draw_networkx_nodes(
    visual_graph,
    positions,
    node_size=450,
    node_color="lightblue",
    edgecolors="black"
)

nx.draw_networkx_edges(
    visual_graph,
    positions,
    edge_color=edge_colors,
    arrows=True,
    arrowsize=15,
    width=1.8,
    alpha=0.75,
    node_size=450
)

# Convert numerical IDs back to account names
labels = {
    node: str(id_to_node[node])
    for node in visual_graph.nodes()
}

nx.draw_networkx_labels(
    visual_graph,
    positions,
    labels=labels,
    font_size=7
)

legend_items = [
    Line2D([0], [0], color=color, lw=3, label=name.title())
    for name, color in typology_colors.items()
]

plt.legend(handles=legend_items, loc="upper right")

plt.title(
    "FlowPrint - Largest AML Connected Component\n"
    f"{visual_graph.number_of_nodes()} Accounts | "
    f"{len(visual_edges)} AML Transactions",
    fontsize=15
)

plt.axis("off")
plt.tight_layout()

# Save image
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "largest_aml_component.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nVisualization saved to:", output_path)
print("Step 2.8 completed successfully!")
