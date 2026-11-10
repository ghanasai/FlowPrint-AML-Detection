
from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parent.parent

df = pd.read_csv(
    ROOT / "AMLNet_v2_transactions.csv",
    usecols=[
        "metadata",
        "isMoneyLaundering"
    ]
)

pattern = re.compile(
    r"""['"]timestamp['"]\s*:\s*
    datetime\.datetime\(
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
    raise ValueError(
        f"Missing timestamps: {df['timestamp'].isna().sum()}"
    )

df = df.sort_values("timestamp").reset_index(drop=True)

n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

for name, part in [
    ("Train", df.iloc[:train_end]),
    ("Validation", df.iloc[train_end:val_end]),
    ("Test", df.iloc[val_end:])
]:
    print(f"\n{name}")
    print("Rows:", len(part))
    print("AML:", int(part["isMoneyLaundering"].sum()))
    print("Normal:", int((part["isMoneyLaundering"] == 0).sum()))
    print("First timestamp:", part["timestamp"].min())
    print("Last timestamp:", part["timestamp"].max())
