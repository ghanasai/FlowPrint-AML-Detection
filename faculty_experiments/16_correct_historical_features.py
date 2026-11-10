
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA = BASE / "data"

SOURCE = ROOT / "AMLNet_v2_transactions.csv"
HISTORY_FILE = DATA / "historical_features.csv"
OUTPUT = DATA / "historical_features_corrected.csv"

FEATURES = [
    "sender_prev_tx_count",
    "receiver_prev_tx_count",
    "sender_prev_total_amount",
    "receiver_prev_total_amount",
    "sender_prev_avg_amount",
    "receiver_prev_avg_amount"
]

print("Loading original transactions...")

transactions = pd.read_csv(
    SOURCE,
    usecols=["nameOrig", "nameDest", "amount"]
)

transactions["original_row_id"] = np.arange(
    len(transactions)
)

print("Loading previously extracted timestamps...")

history = pd.read_csv(
    HISTORY_FILE,
    usecols=["original_row_id", "timestamp"],
    parse_dates=["timestamp"]
)

df = transactions.merge(
    history,
    on="original_row_id",
    how="left",
    validate="one_to_one"
)

assert len(df) == 1_090_000
assert not df["timestamp"].isna().any()

print("Calculating strictly historical features...")

def create_features(frame, account_column, prefix):
    # Aggregate all transactions at each timestamp first.
    # This prevents simultaneous transactions from seeing
    # one another's amounts or counts.
    grouped = (
        frame.groupby(
            [account_column, "timestamp"],
            as_index=False
        )
        .agg(
            tx_count=("amount", "size"),
            tx_amount=("amount", "sum")
        )
        .sort_values([account_column, "timestamp"])
    )

    grouped[f"{prefix}_prev_tx_count"] = (
        grouped.groupby(account_column)["tx_count"].cumsum()
        - grouped["tx_count"]
    )

    grouped[f"{prefix}_prev_total_amount"] = (
        grouped.groupby(account_column)["tx_amount"].cumsum()
        - grouped["tx_amount"]
    )

    grouped[f"{prefix}_prev_avg_amount"] = (
        grouped[f"{prefix}_prev_total_amount"]
        / grouped[f"{prefix}_prev_tx_count"].replace(0, np.nan)
    ).fillna(0)

    columns = [
        account_column,
        "timestamp",
        f"{prefix}_prev_tx_count",
        f"{prefix}_prev_total_amount",
        f"{prefix}_prev_avg_amount"
    ]

    return grouped[columns]

sender_features = create_features(
    df, "nameOrig", "sender"
)

receiver_features = create_features(
    df, "nameDest", "receiver"
)

df = df.merge(
    sender_features,
    on=["nameOrig", "timestamp"],
    how="left",
    validate="many_to_one"
)

df = df.merge(
    receiver_features,
    on=["nameDest", "timestamp"],
    how="left",
    validate="many_to_one"
)

assert not df[FEATURES].isna().any().any()
assert (df[FEATURES] >= -1e-7).all().all()

df[["original_row_id"] + FEATURES].to_csv(
    OUTPUT,
    index=False
)

print("\nCORRECTED HISTORICAL FEATURES")
print("--------------------------------")
print("Rows:", len(df))
print("Features:", len(FEATURES))
print("Missing values:", int(df[FEATURES].isna().sum().sum()))

print("\nSaved:", OUTPUT)
print("\nCorrection completed successfully.")
