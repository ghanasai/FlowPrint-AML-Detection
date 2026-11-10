
from pathlib import Path
import pandas as pd

# Locate the original FlowPrint dataset
BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "AMLNet_v2_transactions.csv"

print("Loading AMLNet v2 dataset...")

df = pd.read_csv(
    DATASET,
    usecols=[
        "nameOrig",
        "nameDest",
        "amount",
        "isMoneyLaundering"
    ]
)

print("\nTotal transactions:", len(df))

# Separate AML and normal transactions
aml = df[df["isMoneyLaundering"] == 1].copy()
normal = df[df["isMoneyLaundering"] == 0].copy()

print("AML transactions:", len(aml))
print("Normal transactions:", len(normal))

# Find all accounts involved in AML transactions
aml_accounts = set(aml["nameOrig"]) | set(aml["nameDest"])

print("\nUnique AML-associated accounts:", len(aml_accounts))

# Find normal transactions touching AML-associated accounts
connected_normal = normal[
    normal["nameOrig"].isin(aml_accounts)
    | normal["nameDest"].isin(aml_accounts)
].copy()

print(
    "Normal transactions directly connected "
    "to AML-associated accounts:",
    len(connected_normal)
)

# Count AML transactions with repeated/linked accounts
sender_counts = aml["nameOrig"].value_counts()
receiver_counts = aml["nameDest"].value_counts()

print("\nAML senders with multiple outgoing transactions:",
      (sender_counts > 1).sum())

print("AML receivers with multiple incoming transactions:",
      (receiver_counts > 1).sum())

# Save only the connectivity candidates for inspection
OUTPUT = Path(__file__).resolve().parent / "results"
OUTPUT.mkdir(exist_ok=True)

connected_normal.to_csv(
    OUTPUT / "connected_normal_candidates.csv",
    index=False
)

print("\nConnectivity check completed!")
