import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import joblib
from pathlib import Path


# ==========================================
# 1. LOAD DATA
# ==========================================

data = np.load("outputs/cremad_features_v1.npz")

X = data["X"]
y = data["y"]
splits = data["splits"]

X_train = X[splits == "train"]
y_train = y[splits == "train"]

X_val = X[splits == "validation"]
y_val = y[splits == "validation"]

X_test = X[splits == "test"]
y_test = y[splits == "test"]

print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ==========================================
# 2. DEFINE SVM CONFIGURATIONS
# ==========================================

configs = [
    {"kernel": "linear", "C": 1},
    {"kernel": "linear", "C": 10},

    {"kernel": "rbf", "C": 1, "gamma": "scale"},
    {"kernel": "rbf", "C": 10, "gamma": "scale"},
    {"kernel": "rbf", "C": 100, "gamma": "scale"},

    {"kernel": "rbf", "C": 10, "gamma": 0.01},
    {"kernel": "rbf", "C": 10, "gamma": 0.001},
]


# ==========================================
# 3. TRAIN AND SELECT USING VALIDATION
# ==========================================

results = []
best_model = None
best_f1 = -1
best_config = None

for config in configs:

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            **config,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    val_predictions = model.predict(X_val)

    val_accuracy = accuracy_score(
        y_val, val_predictions
    )

    val_macro_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro"
    )

    val_weighted_f1 = f1_score(
        y_val,
        val_predictions,
        average="weighted"
    )

    results.append({
        "configuration": str(config),
        "val_accuracy": val_accuracy,
        "val_macro_f1": val_macro_f1,
        "val_weighted_f1": val_weighted_f1
    })

    print("\nConfiguration:", config)
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Macro F1: {val_macro_f1:.4f}")
    print(f"Validation Weighted F1: {val_weighted_f1:.4f}")

    if val_macro_f1 > best_f1:
        best_f1 = val_macro_f1
        best_model = model
        best_config = config


# ==========================================
# 4. CONFIGURATION COMPARISON
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="val_macro_f1",
    ascending=False
)

print("\n" + "=" * 50)
print("VALIDATION RESULTS")
print("=" * 50)

print(results_df.to_string(index=False))

print("\nBest configuration:", best_config)
print("Best validation Macro F1:", best_f1)


# ==========================================
# 5. FINAL TEST EVALUATION
# ==========================================

test_predictions = best_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test, test_predictions
)

test_macro_f1 = f1_score(
    y_test,
    test_predictions,
    average="macro"
)

test_weighted_f1 = f1_score(
    y_test,
    test_predictions,
    average="weighted"
)

print("\n" + "=" * 50)
print("FINAL TEST RESULTS")
print("=" * 50)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"Test Weighted F1: {test_weighted_f1:.4f}")

print("\nClassification Report:")

print(classification_report(
    y_test,
    test_predictions,
    labels=["angry", "happy", "neutral", "sad"],
    zero_division=0
))

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=["angry", "happy", "neutral", "sad"]
)

print(cm)


# ==========================================
# 6. SAVE RESULTS AND MODEL
# ==========================================

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

joblib.dump(
    best_model,
    "models/cremad_svm_v1.pkl"
)

results_df.to_csv(
    "outputs/cremad_svm_validation_results.csv",
    index=False
)

np.save(
    "outputs/cremad_svm_test_predictions.npy",
    test_predictions
)

print("\nModel saved: models/cremad_svm_v1.pkl")
print("Validation results saved.")
print("Test predictions saved.")

print("\nCREMA-D Experiment 1 complete!")

'''
Macro F1 calculates the F1 score for each emotion independently and averages them equally.

That is important because we don't want our model to perform well on Angry, Happy and Sad while failing to recognize Neutral.
'''