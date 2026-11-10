
import pandas as pd
import re
from pathlib import Path

# ==========================================
# FLOWPRINT - PHASE 3
# Step 3.3: Extract Transaction Timestamps
# ==========================================

base_dir = Path(__file__).resolve().parent

input_file = base_dir / "AMLNet_v2_transactions.csv"
output_file = base_dir / "flowprint_timestamps.csv"

print("=" * 55)
print("STEP 3.3 - TIMESTAMP EXTRACTION")
print("=" * 55)

# Read only the metadata column
df = pd.read_csv(
    input_file,
    usecols=["metadata"]
)

print("\nTotal transactions:", len(df))

# Match the timestamp field only
timestamp_pattern = re.compile(
    r"['\"]timestamp['\"]\s*:\s*"
    r"datetime\.datetime\("
    r"(\d{4}),\s*(\d{1,2}),\s*(\d{1,2}),\s*"
    r"(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2})"
    r"(?:,\s*(\d{1,6}))?\)"
)

def extract_timestamp(metadata):
    match = timestamp_pattern.search(str(metadata))

    if match is None:
        return None

    values = list(match.groups())

    # Microseconds are optional
    if values[6] is None:
        values[6] = "0"

    year, month, day, hour, minute, second, microsecond = (
        map(int, values)
    )

    try:
        return pd.Timestamp(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            microsecond=microsecond
        )
    except (ValueError, TypeError):
        return None

print("\nExtracting timestamps...")

timestamps = df["metadata"].map(extract_timestamp)

print("\nSuccessfully extracted:", timestamps.notna().sum())
print("Missing timestamps:", timestamps.isna().sum())

print("\nEarliest timestamp:", timestamps.min())
print("Latest timestamp:", timestamps.max())

print("\nFirst 5 timestamps:")
print(timestamps.head().to_string(index=False))

print("\nTimestamps already sorted:",
      timestamps.is_monotonic_increasing)

# Save timestamps with original transaction row ID
timestamp_table = pd.DataFrame({
    "transaction_id": range(len(timestamps)),
    "timestamp": timestamps
})

timestamp_table.to_csv(output_file, index=False)

print("\nSaved:", output_file)
print("Step 3.3 completed successfully!")
