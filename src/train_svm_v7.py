
import os
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# 1. Paths
# ==========================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# 2. Load datasets
# ==========================================

train = np.load(
    os.path.join(OUTPUT_DIR, "train_data_v7.npz")
)

val = np.load(
    os.path.join(OUTPUT_DIR, "validation_data_v7.npz")
)

test = np.load(
    os.path.join(OUTPUT_DIR, "test_data_v7.npz")
)

X_train = train["X"]
y_train = train["y"]

X_val = val["X"]
y_val = val["y"]

X_test = test["X"]
y_test = test["y"]

print("Datasets loaded!")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ==========================================
# 3. Candidate SVM configurations
# ==========================================

candidates = [
    ("Linear_C1", "linear", 1, None),
    ("Linear_C10", "linear", 10, None),

    ("RBF_C0.1", "rbf", 0.1, "scale"),
    ("RBF_C1", "rbf", 1, "scale"),
    ("RBF_C10", "rbf", 10, "scale"),
    ("RBF_C100", "rbf", 100, "scale"),

    ("RBF_C1_Gamma0.001", "rbf", 1, 0.001),
    ("RBF_C10_Gamma0.001", "rbf", 10, 0.001),

    ("RBF_C1_Gamma0.01", "rbf", 1, 0.01),
    ("RBF_C10_Gamma0.01", "rbf", 10, 0.01),

    ("RBF_C1_Gamma0.1", "rbf", 1, 0.1),
    ("RBF_C10_Gamma0.1", "rbf", 10, 0.1),
]


# ==========================================
# 4. Train and validate candidates
# ==========================================

results = []
trained_models = {}

print("\nStarting SVM model comparison...\n")

for name, kernel, C, gamma in candidates:

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=kernel,
            C=C,
            gamma=gamma if gamma is not None else "scale",
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    # Validation predictions
    val_predictions = model.predict(X_val)

    val_accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    val_macro_f1 = f1_score(
        y_val,
        val_predictions,
        average="macro",
        zero_division=0
    )

    val_weighted_f1 = f1_score(
        y_val,
        val_predictions,
        average="weighted",
        zero_division=0
    )

    results.append({
        "name": name,
        "model": model,
        "accuracy": val_accuracy,
        "macro_f1": val_macro_f1,
        "weighted_f1": val_weighted_f1
    })

    trained_models[name] = model

    print(
        f"{name:25s} | "
        f"Accuracy: {val_accuracy:.4f} | "
        f"Macro F1: {val_macro_f1:.4f} | "
        f"Weighted F1: {val_weighted_f1:.4f}"
    )


# ==========================================
# 5. Select best model
# ==========================================

best_result = max(
    results,
    key=lambda result: result["macro_f1"]
)

best_model = best_result["model"]
best_name = best_result["name"]

print("\n" + "=" * 50)
print("BEST MODEL SELECTED USING VALIDATION MACRO F1")
print("=" * 50)

print("Model:", best_name)
print("Validation Accuracy:", best_result["accuracy"])
print("Validation Macro F1:", best_result["macro_f1"])
print("Validation Weighted F1:", best_result["weighted_f1"])


# ==========================================
# 6. Final test evaluation
# ==========================================

print("\nEvaluating selected model on TEST set...")

y_test_pred = best_model.predict(X_test)

test_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

test_macro_f1 = f1_score(
    y_test,
    y_test_pred,
    average="macro",
    zero_division=0
)

test_weighted_f1 = f1_score(
    y_test,
    y_test_pred,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 50)
print("EXPERIMENT 7 - FINAL TEST RESULTS")
print("=" * 50)

print("Selected model:", best_name)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"Test Weighted F1: {test_weighted_f1:.4f}")


# ==========================================
# 7. Classification report
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_test_pred,
        labels=["angry", "happy", "neutral", "sad"],
        zero_division=0
    )
)


# ==========================================
# 8. Confusion matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_test_pred,
    labels=["angry", "happy", "neutral", "sad"]
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 9. Save best model
# ==========================================

model_path = os.path.join(
    MODEL_DIR,
    "emotion_svm_v7.pkl"
)

joblib.dump(best_model, model_path)

print("\nBest SVM pipeline saved:")
print(model_path)


# ==========================================
# 10. Save experiment results
# ==========================================

result_summary = np.array([
    [
        result["name"],
        result["accuracy"],
        result["macro_f1"],
        result["weighted_f1"]
    ]
    for result in results
], dtype=object)

np.save(
    os.path.join(OUTPUT_DIR, "svm_v7_results.npy"),
    result_summary
)

np.save(
    os.path.join(OUTPUT_DIR, "svm_v7_test_confusion_matrix.npy"),
    cm
)

print("\nExperiment 7 completed!")
