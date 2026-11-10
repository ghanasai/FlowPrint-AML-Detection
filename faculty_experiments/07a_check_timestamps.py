
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

sample = pd.read_csv(
    ROOT / "faculty_experiments" / "data"
    / "connected_sample_163aml_500normal.csv"
)

print("\nTIME INFORMATION CHECK")
print("----------------------")

print("Total transactions:", len(sample))

print("\nUnique step values:")
print(sample["step"].value_counts(dropna=False).head(20))

print("\nNumber of unique steps:")
print(sample["step"].nunique(dropna=False))

print("\nMetadata examples:")
for value in sample["metadata"].head(3):
    print(str(value)[:400])

print("\nHour distribution:")
print(sample["hour"].value_counts().sort_index())

print("\nAML distribution by step:")
print(pd.crosstab(
    sample["step"],
    sample["isMoneyLaundering"]
).head(20))
