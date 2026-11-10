
from pathlib import Path
import time
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

print("STEP 23A: CONTROLLED BASELINE COMPARISON")
print("=" * 65)

ORIGINAL_6 = [
    "amount",
    "oldbalanceOrg",
    "hour",
    "day_of_week",
    "day_of_month",
    "month"
]

HISTORICAL = [
    "sender_prev_tx_count",
    "receiver_prev_tx_count",
    "sender_prev_total_amount",
    "receiver_prev_total_amount",
    "sender_prev_avg_amount",
    "receiver_prev_avg_amount"
]

GRAPH = [
    "sender_out_degree",
    "sender_in_degree",
    "receiver_in_degree",
    "receiver_out_degree",
    "pair_previous_count"
]

FEATURES_17 = ORIGINAL_6 + HISTORICAL + GRAPH

MODELS = {
    "Baseline_6": ORIGINAL_6,
    "FlowPrint_17": FEATURES_17
}

FOLDS = [
    (1, 4, 5, 6),
    (2, 6, 7, 8),
    (3, 8, 9, 10)
]

print("\nLoading original dataset...")

df = pd.read_csv(
    ROOT / "AMLNet_v2_transactions.csv",
    usecols=ORIGINAL_6 + ["isMoneyLaundering"]
)

df["original_row_id"] = np.arange(len(df))

timestamps = pd.read_csv(
    DATA / "historical_features.csv",
    usecols=["original_row_id", "timestamp"],
    parse_dates=["timestamp"]
)

history = pd.read_csv(
    DATA / "historical_features_corrected.csv",
    usecols=["original_row_id"] + HISTORICAL
)

graph = pd.read_csv(
    DATA / "graph_features.csv",
    usecols=["original_row_id"] + GRAPH
)

for name, frame in [
    ("timestamps", timestamps),
    ("history", history),
    ("graph", graph)
]:
    assert frame["original_row_id"].is_unique, (
        f"Duplicate IDs in {name}"
    )

df = df.merge(
    timestamps,
    on="original_row_id",
    validate="one_to_one"
)

df = df.merge(
    history,
    on="original_row_id",
    validate="one_to_one"
)

df = df.merge(
    graph,
    on="original_row_id",
    validate="one_to_one"
)

assert len(df) == 1090000
assert df["timestamp"].notna().all()
assert not df[FEATURES_17].isna().any().any()

df = df.sort_values(
    ["timestamp", "original_row_id"]
).reset_index(drop=True)

df["time_decile"] = (
    np.arange(len(df)) * 10 // len(df)
) + 1

# Check that fold boundaries do not split
# transactions with identical timestamps.
for boundary in [
    i * (len(df) // 10)
    for i in range(1, 10)
]:
    assert (
        df.loc[boundary - 1, "timestamp"]
        < df.loc[boundary, "timestamp"]
    ), "A temporal boundary splits identical timestamps"

def best_threshold(y_true, scores):

    precision, recall, thresholds = (
        precision_recall_curve(y_true, scores)
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12
        )
    )

    idx = int(np.argmax(f1))

    return (
        float(thresholds[idx]),
        float(f1[idx])
    )

results = []

for fold, train_end, val_id, test_id in FOLDS:

    print("\n" + "=" * 65)
    print("FOLD", fold)
    print("=" * 65)

    train = df[df["time_decile"] <= train_end]
    val = df[df["time_decile"] == val_id]
    test = df[df["time_decile"] == test_id]

    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()

    y_train = train["isMoneyLaundering"].astype(int)
    y_val = val["isMoneyLaundering"].astype(int)
    y_test = test["isMoneyLaundering"].astype(int)

    print("Training:", len(train))
    print("Validation:", len(val))
    print("Testing:", len(test))
    print("Test AML:", int(y_test.sum()))

    for name, features in MODELS.items():

        print("\nTraining", name)

        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            min_samples_leaf=5,
            class_weight=None,
            random_state=42,
            n_jobs=-1
        )

        start = time.perf_counter()

        model.fit(train[features], y_train)

        elapsed = time.perf_counter() - start

        val_scores = model.predict_proba(
            val[features]
        )[:, 1]

        threshold, validation_f1 = best_threshold(
            y_val, val_scores
        )

        test_scores = model.predict_proba(
            test[features]
        )[:, 1]

        predictions = (
            test_scores >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).ravel()

        row = {
            "Fold": fold,
            "Model": name,
            "Features": len(features),
            "Train_Size": len(train),
            "Validation_Size": len(val),
            "Test_Size": len(test),
            "Test_AML": int(y_test.sum()),
            "Threshold": threshold,
            "Validation_F1": validation_f1,
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1": f1_score(
                y_test, predictions, zero_division=0
            ),
            "PR_AUC": average_precision_score(
                y_test, test_scores
            ),
            "ROC_AUC": roc_auc_score(
                y_test, test_scores
            ),
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
            "Training_Seconds": round(elapsed, 2)
        }

        results.append(row)

        print("Precision:", round(row["Precision"], 4))
        print("Recall:", round(row["Recall"], 4))
        print("F1:", round(row["F1"], 4))
        print("PR-AUC:", round(row["PR_AUC"], 4))
        print("TP:", tp, "FP:", fp)
        print("FN:", fn, "TN:", tn)

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS / "controlled_baseline_results.csv",
    index=False
)

print("\nFINAL CONTROLLED COMPARISON")
print("=" * 65)

print(
    results_df[
        [
            "Fold", "Model", "Precision",
            "Recall", "F1", "PR_AUC",
            "TP", "FP", "FN"
        ]
    ].to_string(index=False)
)

print("\nMEAN PERFORMANCE")

summary = (
    results_df.groupby("Model")[
        ["Precision", "Recall", "F1", "PR_AUC"]
    ]
    .agg(["mean", "std"])
)

print(summary.to_string())

print("\nSTEP 23A COMPLETED SUCCESSFULLY")
