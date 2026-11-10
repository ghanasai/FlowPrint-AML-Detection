
import pandas as pd
from pathlib import Path

# ==========================================
# FLOWPRINT - MONEY LAUNDERING DETECTION
# Step 1.7: Dataset Loading
# ==========================================

print("=" * 55)
print("FLOWPRINT - MONEY LAUNDERING DETECTION")
print("=" * 55)

# Dataset location
file_path = Path(__file__).resolve().parent / "AMLNet_v2_transactions.csv"

# Check whether the file exists
if not file_path.exists():
    print("ERROR: Dataset not found!")
    print("Expected location:", file_path)

else:
    print("\nLoading AMLNet v2 dataset...")

    # Read CSV
    df = pd.read_csv(file_path)

    print("\nDataset loaded successfully!")

    # Dataset dimensions
    print("\nTotal transactions:", len(df))
    print("Total columns:", len(df.columns))

    # Display column names
    print("\nDataset columns:")
    print(df.columns.tolist())

    # Display first five transactions
    print("\nFirst 5 transactions:")
    print(df.head().to_string(index=False))



# ==========================================
# Step 1.8: Dataset Audit
# ==========================================

if file_path.exists():

    print("\n" + "=" * 55)
    print("STEP 1.8 - DATASET AUDIT")
    print("=" * 55)

    # 1. Dataset shape
    print("\nDataset shape:")
    print(df.shape)

    # 2. Column data types
    print("\nColumn data types:")
    print(df.dtypes)

    # 3. Missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # 4. Normal vs money laundering transactions
    print("\nMoney laundering distribution:")
    print(df["isMoneyLaundering"].value_counts())

    # 5. AML percentage
    aml_percentage = df["isMoneyLaundering"].mean() * 100

    print("\nMoney laundering percentage:")
    print(round(aml_percentage, 4), "%")

    # 6. Laundering typologies
    print("\nLaundering typologies:")
    print(df["laundering_typology"].value_counts())

    # 7. Transaction types
    print("\nTransaction types:")
    print(df["type"].value_counts())

    # 8. Check possible target leakage
    print("\nFraud vs money laundering labels:")
    print(
        pd.crosstab(
            df["isFraud"],
            df["isMoneyLaundering"]
        )
    )

    print("\nDataset audit completed!")
