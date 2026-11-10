
from pathlib import Path
import pandas as pd
import networkx as nx

BASE = Path(__file__).resolve().parent

sample = pd.read_csv(
    BASE / "data" / "connected_sample_163aml_500normal.csv"
)

# Build graph using all sampled transactions
G = nx.Graph()

G.add_edges_from(
    sample[["nameOrig", "nameDest"]].itertuples(
        index=False, name=None
    )
)

components = sorted(
    nx.connected_components(G),
    key=len,
    reverse=True
)

print("\nSAMPLED NETWORK VALIDATION")
print("--------------------------")

print("Total transactions:", len(sample))
print("Unique accounts:", G.number_of_nodes())
print("Unique account connections:", G.number_of_edges())
print("Connected components:", len(components))

print("\nTop 10 connected components:")

summary = []

for i, nodes in enumerate(components, start=1):
    transactions = sample[
        sample["nameOrig"].isin(nodes)
        & sample["nameDest"].isin(nodes)
    ]

    aml_count = int(
        transactions["isMoneyLaundering"].sum()
    )

    normal_count = len(transactions) - aml_count

    summary.append({
        "component": i,
        "accounts": len(nodes),
        "transactions": len(transactions),
        "aml": aml_count,
        "normal": normal_count
    })

    if i <= 10:
        print(
            f"Network {i}: "
            f"{len(nodes)} accounts, "
            f"{len(transactions)} transactions, "
            f"{aml_count} AML, "
            f"{normal_count} normal"
        )

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    BASE / "results" / "sample_network_validation.csv",
    index=False
)

print("\nTotal AML across components:",
      summary_df["aml"].sum())

print("Total normal across components:",
      summary_df["normal"].sum())

print("\nValidation complete!")
