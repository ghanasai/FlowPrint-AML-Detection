
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "results"

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

validation = pd.read_csv(
    DATA / "comparison_validation.csv"
)

test = pd.read_csv(
    DATA / "comparison_test.csv"
)

X_val = validation[FEATURES].fillna(0)
y_val = validation["isMoneyLaundering"].astype(int)

X_test = test[FEATURES].fillna(0)
y_test = test["isMoneyLaundering"].astype(int)

models = {
    "Connected Sample": OUT / "connected_sample_rf.joblib",
    "Original Dataset": OUT / "original_dataset_rf.joblib"
}

results = []

for name, path in models.items():

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)

    model = joblib.load(path)

    val_probs = model.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = (
        precision_recall_curve(y_val, val_probs)
    )

    # Last precision/recall point has no threshold.
    f1_values = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12
        )
    )

    best_index = int(np.argmax(f1_values))
    best_threshold = float(thresholds[best_index])

    print("Best validation threshold:", best_threshold)
    print("Best validation F1:", f1_values[best_index])

    test_probs = model.predict_proba(X_test)[:, 1]
    test_pred = (test_probs >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, test_pred, labels=[0, 1]
    ).ravel()

    row = {
        "Model": name,
        "Threshold": best_threshold,
        "Precision": precision_score(
            y_test, test_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_test, test_pred, zero_division=0
        ),
        "F1": f1_score(
            y_test, test_pred, zero_division=0
        ),
        "PR-AUC": average_precision_score(
            y_test, test_probs
        ),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn)
    }

    results.append(row)

    print("\nOPTIMIZED TEST RESULTS")

    for key, value in row.items():
        print(f"{key}: {value}")

comparison = pd.DataFrame(results)

comparison.to_csv(
    OUT / "optimized_threshold_comparison.csv",
    index=False
)

print("\n" + "=" * 55)
print("FINAL OPTIMIZED COMPARISON")
print("=" * 55)
print(comparison.to_string(index=False))

print("\nSaved:")
print(OUT / "optimized_threshold_comparison.csv")
