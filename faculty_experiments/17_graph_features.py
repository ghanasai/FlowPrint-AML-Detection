
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

SOURCE = ROOT / "AMLNet_v2_transactions.csv"
HISTORY = DATA / "historical_features.csv"
OUTPUT = DATA / "graph_features.csv"

FEATURES = [
    "sender_out_degree",
    "sender_in_degree",
    "receiver_in_degree",
    "receiver_out_degree",
    "pair_previous_count"
]

print("STEP 16A: HISTORICAL GRAPH FEATURES")
print("=" * 55)

print("\nLoading transactions...")

df = pd.read_csv(
    SOURCE,
    usecols=["nameOrig", "nameDest"]
)

df["original_row_id"] = np.arange(len(df))

timestamps = pd.read_csv(
    HISTORY,
    usecols=["original_row_id", "timestamp"],
    parse_dates=["timestamp"]
)

df = df.merge(
    timestamps,
    on="original_row_id",
    validate="one_to_one"
)

assert len(df) == 1_090_000
assert not df["timestamp"].isna().any()

print("Sorting transactions chronologically...")

df = df.sort_values(
    ["timestamp", "original_row_id"]
).reset_index(drop=True)

# Each unique directed account pair first becomes
# an existing graph edge at its earliest timestamp.
print("Finding first occurrence of account connections...")

first_edges = (
    df.groupby(["nameOrig", "nameDest"], as_index=False)
    ["timestamp"].min()
)

# Count unique edges that existed STRICTLY BEFORE
# each transaction timestamp.
def previous_unique_degree(
    edge_table, account_col, prefix
):
    events = (
        edge_table.groupby(
            [account_col, "timestamp"],
            as_index=False
        )
        .size()
        .rename(columns={"size": "new_edges"})
        .sort_values([account_col, "timestamp"])
    )

    events["previous"] = (
        events.groupby(account_col)["new_edges"].cumsum()
        - events["new_edges"]
    )

    # Query prior edge counts using backward as-of join.
    queries = (
        df[["original_row_id", "timestamp", account_col]]
        .sort_values("timestamp")
    )

    events = events.sort_values("timestamp")

    result = pd.merge_asof(
        queries,
        events[[account_col, "timestamp", "previous", "new_edges"]],
        on="timestamp",
        by=account_col,
        direction="backward",
        allow_exact_matches=True
    )

    # When timestamps match, "previous" excludes
    # edges first observed at that timestamp.
    # When timestamps differ, include all edges
    # first observed at the earlier event timestamp.
    exact = (
        result["timestamp"].eq(
            pd.merge_asof(
                queries,
                events[[account_col, "timestamp"]].rename(
                    columns={"timestamp": "event_timestamp"}
                ).sort_values("event_timestamp"),
                left_on="timestamp",
                right_on="event_timestamp",
                by=account_col,
                direction="backward"
            )["event_timestamp"]
        )
    )

    counts = result["previous"].fillna(0)
    additions = result["new_edges"].fillna(0)

    result[prefix] = np.where(
        exact,
        counts,
        counts + additions
    ).astype("int32")

    return result[["original_row_id", prefix]]

# Simpler exact and earlier timestamp implementation:
# Use cumulative event counts and a strict as-of join.
def degree_feature(edge_table, account_col, output_col):
    events = (
        edge_table.groupby(
            [account_col, "timestamp"],
            as_index=False
        )
        .size()
        .sort_values([account_col, "timestamp"])
    )

    events["cumulative"] = (
        events.groupby(account_col)["size"].cumsum()
    )

    queries = df[
        ["original_row_id", "timestamp", account_col]
    ].sort_values("timestamp")

    events = events.sort_values("timestamp")

    joined = pd.merge_asof(
        queries,
        events[[account_col, "timestamp", "cumulative"]],
        on="timestamp",
        by=account_col,
        direction="backward",
        allow_exact_matches=False
    )

    joined[output_col] = (
        joined["cumulative"]
        .fillna(0)
        .astype("int32")
    )

    return joined[["original_row_id", output_col]]

print("Calculating sender out-degree...")
sender_out = degree_feature(
    first_edges, "nameOrig", "sender_out_degree"
)

print("Calculating receiver in-degree...")
receiver_in = degree_feature(
    first_edges, "nameDest", "receiver_in_degree"
)



print("Calculating sender in-degree...")

# Use destination accounts from original edges
sender_in_events = pd.DataFrame({
    "nameOrig": first_edges["nameDest"].to_numpy(),
    "timestamp": first_edges["timestamp"].to_numpy()
})

print("Sender in-degree columns:",
      sender_in_events.columns.tolist())

assert sender_in_events.columns.is_unique

sender_in = degree_feature(
    sender_in_events,
    "nameOrig",
    "sender_in_degree"
)

print("Calculating receiver out-degree...")

# Use source accounts from original edges
receiver_out_events = pd.DataFrame({
    "nameDest": first_edges["nameOrig"].to_numpy(),
    "timestamp": first_edges["timestamp"].to_numpy()
})

print("Receiver out-degree columns:",
      receiver_out_events.columns.tolist())

assert receiver_out_events.columns.is_unique

receiver_out = degree_feature(
    receiver_out_events,
    "nameDest",
    "receiver_out_degree"
)



print("Calculating previous directed-pair transfers...")

pair_events = (
    df.groupby(
        ["nameOrig", "nameDest", "timestamp"],
        as_index=False
    )
    .size()
    .sort_values(["nameOrig", "nameDest", "timestamp"])
)

pair_events["previous_count"] = (
    pair_events.groupby(["nameOrig", "nameDest"])["size"]
    .cumsum()
    - pair_events["size"]
)

pair_events = pair_events[
    ["nameOrig", "nameDest", "timestamp", "previous_count"]
]

df = df.merge(
    pair_events,
    on=["nameOrig", "nameDest", "timestamp"],
    how="left",
    validate="many_to_one"
)

df["pair_previous_count"] = (
    df["previous_count"].astype("int32")
)

print("Combining features...")

for frame in [
    sender_out, sender_in, receiver_in, receiver_out
]:
    df = df.merge(
        frame,
        on="original_row_id",
        validate="one_to_one"
    )

assert not df[FEATURES].isna().any().any()
assert (df[FEATURES] >= 0).all().all()

df[["original_row_id"] + FEATURES].to_csv(
    OUTPUT,
    index=False
)

print("\nGRAPH FEATURE ENGINEERING COMPLETE")
print("----------------------------------")
print("Transactions:", len(df))
print("Features:", len(FEATURES))
print("Missing values:", int(df[FEATURES].isna().sum().sum()))

print("\nFeature statistics:")
print(df[FEATURES].describe().to_string())

print("\nSaved:", OUTPUT)
