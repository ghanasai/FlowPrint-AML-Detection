
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, roc_auc_score,
    confusion_matrix, precision_recall_curve,
    ConfusionMatrixDisplay
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = BASE / "results"
OUT.mkdir(exist_ok=True)

FEATURES = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

TARGET = "isMoneyLaundering"

print("Loading comparison datasets...")

connected = pd.read_csv(
    DATA / "comparison_connected_train.csv"
)

original = pd.read_csv(
    DATA / "comparison_train.csv"
)

test = pd.read_csv(
    DATA / "comparison_test.csv"
)

# Confirm no training transactions appear in test.
test_ids = set(test["original_row_id"])

for name, train in [
    ("Connected", connected),
    ("Original", original)
]:
    overlap = set(train["original_row_id"]) & test_ids
    if overlap:
        raise ValueError(
            f"{name} training overlaps test: {len(overlap)}"
        )

X_test = test[FEATURES].fillna(0)
y_test = test[TARGET].astype(int)

models = {}
probabilities = {}
predictions = {}
results = []

def train_and_evaluate(name, train):
    print(f"\n{'=' * 55}")
    print(f"TRAINING: {name}")
    print(f"{'=' * 55}")

    X_train = train[FEATURES].fillna(0)
    y_train = train[TARGET].astype(int)

    print("Training records:", len(train))
    print("AML records:", int(y_train.sum()))
    print("Normal records:", int((y_train == 0).sum()))

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight=None
    )

    start = time.perf_counter()
    model.fit(X_train, y_train)
    training_time = time.perf_counter() - start

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "Model": name,
        "Training records": len(train),
        "Training time (s)": round(training_time, 2),
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(
            y_test, preds, zero_division=0
        ),
        "Recall": recall_score(
            y_test, preds, zero_division=0
        ),
        "F1": f1_score(
            y_test, preds, zero_division=0
        ),
        "PR-AUC": average_precision_score(
            y_test, probs
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probs
        )
    }

    cm = confusion_matrix(
        y_test, preds, labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()
    metrics.update({
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    })

    print("\nTEST RESULTS")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nConfusion matrix:")
    print(cm)

    models[name] = model
    probabilities[name] = probs
    predictions[name] = preds
    results.append(metrics)

    joblib.dump(
        model,
        OUT / f"{name.lower().replace(' ', '_')}_rf.joblib"
    )

train_and_evaluate(
    "Connected Sample", connected
)

train_and_evaluate(
    "Original Dataset", original
)

# Save metric comparison
comparison = pd.DataFrame(results)
comparison.to_csv(
    OUT / "final_model_comparison.csv",
    index=False
)

# Save test predictions
pred_df = pd.DataFrame({
    "original_row_id": test["original_row_id"],
    "actual": y_test,
    "connected_probability": probabilities["Connected Sample"],
    "connected_prediction": predictions["Connected Sample"],
    "original_probability": probabilities["Original Dataset"],
    "original_prediction": predictions["Original Dataset"]
})

pred_df.to_csv(
    OUT / "final_comparison_predictions.csv",
    index=False
)

# Plot 1: Key metrics
metric_names = [
    "Precision", "Recall", "F1", "PR-AUC", "ROC-AUC"
]

x = np.arange(len(metric_names))
width = 0.36

fig, ax = plt.subplots(figsize=(11, 6))

for i, row in comparison.iterrows():
    values = [row[m] for m in metric_names]
    ax.bar(
        x + (i - 0.5) * width,
        values,
        width,
        label=row["Model"]
    )

ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Score")
ax.set_title("FlowPrint: Random Forest Model Comparison")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig(
    OUT / "final_metrics_comparison.png",
    dpi=300
)
plt.close(fig)

# Plot 2: Precision-recall curves
fig, ax = plt.subplots(figsize=(9, 6))

for name, probs in probabilities.items():
    precision, recall, _ = precision_recall_curve(
        y_test, probs
    )
    ap = average_precision_score(y_test, probs)
    ax.plot(
        recall,
        precision,
        label=f"{name} (AP={ap:.4f})"
    )

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Common Test Set")
ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(
    OUT / "final_precision_recall.png",
    dpi=300
)
plt.close(fig)

# Plot 3: Confusion matrices
for name, preds in predictions.items():
    cm = confusion_matrix(
        y_test, preds, labels=[0, 1]
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "AML"]
    ).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d"
    )

    ax.set_title(name)
    fig.tight_layout()

    filename = (
        name.lower().replace(" ", "_")
        + "_confusion_matrix.png"
    )

    fig.savefig(OUT / filename, dpi=300)
    plt.close(fig)

print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print("=" * 55)

print(comparison.to_string(index=False))

print("\nSaved results to:")
print(OUT)

print("\nExperiment complete!")
