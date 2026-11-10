
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent

df = pd.read_csv(
    BASE / "data" / "connected_sample_163aml_500normal.csv"
)

print("\nSAMPLE SPLIT AUDIT")
print("------------------")
print("Columns:", df.columns.tolist())
print("Rows:", len(df))

print("\nRepeated directed account pairs:")
print(
    df.duplicated(
        subset=["nameOrig", "nameDest"],
        keep=False
    ).sum()
)

print("\nPossible time columns:")
time_columns = [
    c for c in df.columns
    if any(word in c.lower()
           for word in ["time", "date", "step", "hour"])
]
print(time_columns)

print("\nAML labels:")
print(df["isMoneyLaundering"].value_counts())

if "original_row_id" in df.columns:
    print("\nOriginal row IDs:")
    print("Minimum:", df["original_row_id"].min())
    print("Maximum:", df["original_row_id"].max())
    print(
        "Unique:",
        df["original_row_id"].nunique()
    )
