
from pathlib import Path
import re
import pandas as pd
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
BASE = Path(__file__).resolve().parent
OUT = BASE / "data"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(ROOT / "AMLNet_v2_transactions.csv")

pattern = re.compile(
    r"""['"]timestamp['"]\s*:\s*datetime\.datetime\(
    \s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,
    \s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)
    (?:\s*,\s*(\d+))?\s*\)""",
    re.VERBOSE
)

def extract_timestamp(value):
    match = pattern.search(str(value))
    if not match:
        return pd.NaT

    parts = [
        int(x) if x is not None else 0
        for x in match.groups()
    ]

    try:
        return pd.Timestamp(*parts)
    except (ValueError, TypeError):
        return pd.NaT

df["timestamp"] = df["metadata"].map(extract_timestamp)

if df["timestamp"].isna().any():
    raise ValueError("Some timestamps could not be extracted.")

df["original_row_id"] = range(len(df))

df = df.sort_values(
    ["timestamp", "original_row_id"]
).reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end].copy()
val = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

# Save identical partitions for both models
train.to_csv(OUT / "comparison_train.csv", index=False)
val.to_csv(OUT / "comparison_validation.csv", index=False)
test.to_csv(OUT / "comparison_test.csv", index=False)

# Identify AML connected components in TRAIN only
aml = train[train["isMoneyLaundering"] == 1].copy()
normal = train[train["isMoneyLaundering"] == 0].copy()

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

# Select complete AML components, aiming for 163 rows
component_data = []

for nodes in components:
    rows = aml[
        aml["nameOrig"].isin(nodes)
        & aml["nameDest"].isin(nodes)
    ]
    component_data.append(rows)

# Dynamic programming: find complete components
# totaling exactly 163 AML transactions.
target = 163
dp = {0: []}

for i, rows in enumerate(component_data):
    size = len(rows)
    for total, indices in list(dp.items()):
        new_total = total + size
        if new_total <= target and new_total not in dp:
            dp[new_total] = indices + [i]

if target not in dp:
    raise ValueError(
        "Could not select exactly 163 AML rows "
        "using complete connected components."
    )

selected_aml = pd.concat(
    [component_data[i] for i in dp[target]],
    ignore_index=True
)

aml_accounts = (
    set(selected_aml["nameOrig"])
    | set(selected_aml["nameDest"])
)

connected_normal = normal[
    normal["nameOrig"].isin(aml_accounts)
    | normal["nameDest"].isin(aml_accounts)
]

if len(connected_normal) < 500:
    raise ValueError(
        f"Only {len(connected_normal)} connected "
        "normal transactions available."
    )

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
    OUT / "comparison_connected_train.csv",
    index=False
)

# Validate row separation
train_ids = set(train["original_row_id"])
val_ids = set(val["original_row_id"])
test_ids = set(test["original_row_id"])
sample_ids = set(sample["original_row_id"])

assert sample_ids.issubset(train_ids)
assert not sample_ids.intersection(val_ids)
assert not sample_ids.intersection(test_ids)

print("\nCOMPARISON DATA READY")
print("---------------------")
print("Original training rows:", len(train))
print("Validation rows:", len(val))
print("Test rows:", len(test))
print("Connected sample rows:", len(sample))
print("Sample AML:", int(sample["isMoneyLaundering"].sum()))
print("Sample normal:",
      int((sample["isMoneyLaundering"] == 0).sum()))

print("\nLeakage check:")
print("Sample overlaps validation:",
      len(sample_ids & val_ids))
print("Sample overlaps test:",
      len(sample_ids & test_ids))

print("\nSaved comparison datasets.")
