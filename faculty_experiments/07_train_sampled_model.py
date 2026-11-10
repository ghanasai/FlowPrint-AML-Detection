
from pathlib import Path
import ast
import re
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "connected_sample_163aml_500normal.csv"
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# Extract timestamps from metadata
# Metadata contains datetime.datetime(...) expressions.
pattern = re.compile(
    r"""['"]timestamp['"]\s*:\s*datetime\.datetime\(
    \s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,
    \s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)
    (?:\s*,\s*(\d+))?
    \s*\)""",
    re.VERBOSE
)

def extract_timestamp(value):
    match = pattern.search(str(value))

    if not match:
        return pd.NaT

    parts = [int(x) if x is not None else 0
             for x in match.groups()]

    try:
        return pd.Timestamp(
            year=parts[0],
            month=parts[1],
            day=parts[2],
            hour=parts[3],
            minute=parts[4],
            second=parts[5],
            microsecond=parts[6]
        )
    except ValueError:
        return pd.NaT

df["timestamp"] = df["metadata"].apply(extract_timestamp)

if df["timestamp"].isna().any():
    raise ValueError(
        f"Missing timestamps: {df['timestamp'].isna().sum()}"
    )

features = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

# Sort chronologically
df = df.sort_values(
    ["timestamp", "original_row_id"]
).reset_index(drop=True)

print("\nTIMESTAMP RANGE")
print("Earliest:", df["timestamp"].min())
print("Latest:", df["timestamp"].max())

# Chronological 70/15/15 split
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train = df.iloc[:train_end].copy()
val = df.iloc[train_end:val_end].copy()
test = df.iloc[val_end:].copy()

print("\nCHRONOLOGICAL SPLIT")
print("-------------------")

for name, part in [
    ("Train", train),
    ("Validation", val),
    ("Test", test)
]:
    print(
        name,
        "rows:", len(part),
        "AML:", int(part["isMoneyLaundering"].sum()),
        "Normal:",
        int((part["isMoneyLaundering"] == 0).sum())
    )

for name, part in [
    ("Train", train),
    ("Validation", val),
    ("Test", test)
]:
    if part["isMoneyLaundering"].nunique() < 2:
        raise ValueError(
            f"{name} contains only one class."
        )

X_train = train[features].fillna(0)
y_train = train["isMoneyLaundering"]

X_val = val[features].fillna(0)
y_val = val["isMoneyLaundering"]

X_test = test[features].fillna(0)
y_test = test["isMoneyLaundering"]

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=14,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

def evaluate(name, X, y, rows):
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    print(f"\n{name.upper()} RESULTS")
    print("-----------------------")

    print("Accuracy:", round(
        accuracy_score(y, predictions), 4))
    print("Precision:", round(
        precision_score(y, predictions, zero_division=0), 4))
    print("Recall:", round(
        recall_score(y, predictions, zero_division=0), 4))
    print("F1-score:", round(
        f1_score(y, predictions, zero_division=0), 4))

    print("PR-AUC:", round(
        average_precision_score(y, probabilities), 4))
    print("ROC-AUC:", round(
        roc_auc_score(y, probabilities), 4))

    print("\nConfusion matrix:")
    print(confusion_matrix(y, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(classification_report(
        y, predictions,
        zero_division=0
    ))

    results = pd.DataFrame({
        "original_row_id": rows["original_row_id"].values,
        "actual": y.values,
        "predicted": predictions,
        "probability": probabilities
    })

    results.to_csv(
        OUT / f"sampled_{name.lower()}_predictions.csv",
        index=False
    )

evaluate("Validation", X_val, y_val, val)
evaluate("Test", X_test, y_test, test)

print("\nTraining experiment complete!")
