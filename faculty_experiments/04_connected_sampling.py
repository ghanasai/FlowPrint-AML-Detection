
from pathlib import Path
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(
    ROOT / "AMLNet_v2_transactions.csv"
).copy()

required = [
    "nameOrig", "nameDest",
    "isMoneyLaundering"
]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

# Preserve original row identity
df["original_row_id"] = range(len(df))

aml = df[df["isMoneyLaundering"] == 1].copy()
normal = df[df["isMoneyLaundering"] == 0].copy()

# Build an undirected graph to identify AML components
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

# Select complete components until we reach >=163 AML
selected_ids = set()
selected_accounts = set()

for nodes in components:
    component_aml = aml[
        aml["nameOrig"].isin(nodes)
        & aml["nameDest"].isin(nodes)
    ]

    selected_ids.update(component_aml["original_row_id"])
    selected_accounts.update(nodes)

    if len(selected_ids) >= 163:
        break

selected_aml = aml[
    aml["original_row_id"].isin(selected_ids)
].copy()

# Keep exactly 163 AML records while prioritizing
# complete relationships from the largest components.
selected_aml = selected_aml.head(163).copy()

# Recalculate accounts actually present in selected AML
selected_accounts = (
    set(selected_aml["nameOrig"])
    | set(selected_aml["nameDest"])
)

# Select normal transactions linked to those accounts
connected_normal = normal[
    normal["nameOrig"].isin(selected_accounts)
    | normal["nameDest"].isin(selected_accounts)
].copy()

if len(connected_normal) < 500:
    raise ValueError(
        f"Only {len(connected_normal)} connected "
        "normal transactions found."
    )

# Deterministic random sample from connected candidates
selected_normal = connected_normal.sample(
    n=500,
    random_state=42
)

sample = pd.concat(
    [selected_aml, selected_normal],
    ignore_index=True
)

sample = sample.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

sample.to_csv(
    OUT / "connected_sample_163aml_500normal.csv",
    index=False
)

# Verify account overlap
aml_accounts = (
    set(selected_aml["nameOrig"])
    | set(selected_aml["nameDest"])
)

normal_connected = (
    selected_normal["nameOrig"].isin(aml_accounts)
    | selected_normal["nameDest"].isin(aml_accounts)
)

print("\nCONNECTED SAMPLING RESULTS")
print("--------------------------")
print("AML transactions:", len(selected_aml))
print("Normal transactions:", len(selected_normal))
print("Total:", len(sample))
print("Normal transactions linked to AML accounts:",
      int(normal_connected.sum()))

print("\nClass distribution:")
print(sample["isMoneyLaundering"].value_counts())

print("\nSaved to:")
print(OUT / "connected_sample_163aml_500normal.csv")
