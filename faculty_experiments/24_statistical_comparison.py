
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

print("STEP 21A: STATISTICAL MODEL COMPARISON")
print("=" * 60)

ORIGINAL = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "hour", "day_of_week", "day_of_month", "month"
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

FEATURES_18 = ORIGINAL + HISTORICAL + GRAPH

FEATURES_17 = [
    f for f in FEATURES_18
    if f != "newbalanceOrig"
]

print("\nLoading test dataset...")

test = pd.read_csv(DATA / "comparison_test.csv")

history = pd.read_csv(
    DATA / "historical_features_corrected.csv",
    usecols=["original_row_id"] + HISTORICAL
)

graph = pd.read_csv(
    DATA / "graph_features.csv",
    usecols=["original_row_id"] + GRAPH
)

for name, frame in [
    ("test", test),
    ("history", history),
    ("graph", graph)
]:
    assert frame["original_row_id"].is_unique, (
        f"Duplicate IDs in {name}"
    )

test = test.merge(
    history,
    on="original_row_id",
    validate="one_to_one"
)

test = test.merge(
    graph,
    on="original_row_id",
    validate="one_to_one"
)

assert len(test) == 163500
assert not test[FEATURES_18].isna().any().any()

y = test["isMoneyLaundering"].astype(int).to_numpy()

print("Test transactions:", len(y))
print("AML transactions:", int(y.sum()))

print("\nLoading trained models...")

model_17 = joblib.load(
    RESULTS / "rf_17_feature_model.joblib"
)

model_18 = joblib.load(
    RESULTS / "graph_random_forest_18_features.joblib"
)

# Thresholds selected previously using validation data.
threshold_17 = 0.23755805600307625
threshold_18 = 0.23443083292747288

print("\nGenerating paired predictions...")

prob_17 = model_17.predict_proba(
    test[FEATURES_17]
)[:, 1]

prob_18 = model_18.predict_proba(
    test[FEATURES_18]
)[:, 1]

pred_17 = (prob_17 >= threshold_17).astype(int)
pred_18 = (prob_18 >= threshold_18).astype(int)

print("\nChecking original metrics...")

for name, prob, pred in [
    ("17-feature", prob_17, pred_17),
    ("18-feature", prob_18, pred_18)
]:
    print(
        name,
        "F1:", round(f1_score(y, pred), 6),
        "PR-AUC:", round(
            average_precision_score(y, prob), 6
        )
    )

# Stratified paired bootstrap:
# Sample AML and normal transactions separately.
# Both models use the same sampled indices.
#
# This estimates conditional uncertainty on the
# existing test population. It does not replace
# independent temporal validation.

print("\nRunning stratified paired bootstrap...")

rng = np.random.default_rng(42)
N_BOOTSTRAP = 1000

positive_idx = np.flatnonzero(y == 1)
negative_idx = np.flatnonzero(y == 0)

records = []

for iteration in range(N_BOOTSTRAP):

    sampled_pos = rng.choice(
        positive_idx,
        size=len(positive_idx),
        replace=True
    )

    sampled_neg = rng.choice(
        negative_idx,
        size=len(negative_idx),
        replace=True
    )

    indices = np.concatenate([
        sampled_pos,
        sampled_neg
    ])

    y_boot = y[indices]

    f1_17 = f1_score(
        y_boot, pred_17[indices],
        zero_division=0
    )

    f1_18 = f1_score(
        y_boot, pred_18[indices],
        zero_division=0
    )

    ap_17 = average_precision_score(
        y_boot, prob_17[indices]
    )

    ap_18 = average_precision_score(
        y_boot, prob_18[indices]
    )

    records.append({
        "Iteration": iteration + 1,
        "F1_17": f1_17,
        "F1_18": f1_18,
        "F1_Difference_17_minus_18": f1_17 - f1_18,
        "AP_17": ap_17,
        "AP_18": ap_18,
        "AP_Difference_17_minus_18": ap_17 - ap_18
    })

bootstrap = pd.DataFrame(records)

bootstrap.to_csv(
    RESULTS / "statistical_bootstrap_results.csv",
    index=False
)

print("\n95% BOOTSTRAP CONFIDENCE INTERVALS")
print("-" * 60)

summary_rows = []

columns = [
    "F1_17",
    "F1_18",
    "F1_Difference_17_minus_18",
    "AP_17",
    "AP_18",
    "AP_Difference_17_minus_18"
]

for column in columns:

    values = bootstrap[column].to_numpy()

    lower, upper = np.percentile(
        values, [2.5, 97.5]
    )

    mean = values.mean()

    summary_rows.append({
        "Metric": column,
        "Bootstrap_Mean": mean,
        "CI_Lower": lower,
        "CI_Upper": upper
    })

    print(
        f"{column}: "
        f"Mean={mean:.4f}, "
        f"95% CI=[{lower:.4f}, {upper:.4f}]"
    )

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    RESULTS / "statistical_comparison_summary.csv",
    index=False
)

print("\nPAIRED PREDICTION AGREEMENT")
print("-" * 60)

both_correct = int(
    ((pred_17 == y) & (pred_18 == y)).sum()
)

only_17_correct = int(
    ((pred_17 == y) & (pred_18 != y)).sum()
)

only_18_correct = int(
    ((pred_17 != y) & (pred_18 == y)).sum()
)

both_wrong = int(
    ((pred_17 != y) & (pred_18 != y)).sum()
)

print("Both correct:", both_correct)
print("Only 17-feature correct:", only_17_correct)
print("Only 18-feature correct:", only_18_correct)
print("Both wrong:", both_wrong)

print("\nSTEP 21A COMPLETED SUCCESSFULLY")
