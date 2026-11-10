
import pandas as pd
from pathlib import Path

# ==========================================
# FLOWPRINT - PHASE 4
# Step 4.5: Transaction Amount Audit
# ==========================================

base_dir = Path(__file__).resolve().parent
data_dir = base_dir / "data"

print("=" * 55)
print("STEP 4.5 - AMOUNT DISTRIBUTION AUDIT")
print("=" * 55)

columns = [
    "amount",
    "isMoneyLaundering",
    "laundering_typology"
]

train_df = pd.read_csv(
    data_dir / "train.csv",
    usecols=columns
)

normal = train_df[
    train_df["isMoneyLaundering"] == 0
]["amount"]

aml = train_df[
    train_df["isMoneyLaundering"] == 1
]["amount"]

print("\nNORMAL TRANSACTION AMOUNTS")
print(normal.describe(
    percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99]
).to_string())

print("\nAML TRANSACTION AMOUNTS")
print(aml.describe(
    percentiles=[0.01, 0.25, 0.5, 0.75, 0.95, 0.99]
).to_string())

print("\nAML TYPOLOGY DISTRIBUTION")
print(
    train_df.loc[
        train_df["isMoneyLaundering"] == 1,
        "laundering_typology"
    ].value_counts(dropna=False)
)

print("\nAMOUNTS BY AML TYPOLOGY")

aml_df = train_df[
    train_df["isMoneyLaundering"] == 1
]

print(
    aml_df.groupby("laundering_typology")["amount"]
    .describe()
    .to_string()
)

print("\nPERCENTAGE ABOVE AMOUNT THRESHOLDS")

for threshold in [100, 1000, 10000, 50000, 100000]:
    normal_pct = (normal >= threshold).mean() * 100
    aml_pct = (aml >= threshold).mean() * 100

    print(
        f"Amount >= {threshold:>6}: "
        f"Normal={normal_pct:6.2f}% | "
        f"AML={aml_pct:6.2f}%"
    )

print("\nStep 4.5 completed successfully!")
