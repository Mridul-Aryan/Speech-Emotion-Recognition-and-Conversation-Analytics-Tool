import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import joblib


# ==========================================
# 1. REPRODUCIBILITY
# ==========================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================
# 2. LOAD FEATURE DATA
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
# 3. LABEL ENCODING
# ==========================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

num_classes = len(label_encoder.classes_)

print("\nClasses:", label_encoder.classes_)


# ==========================================
# 4. FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. CLASS WEIGHTS
# ==========================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_encoded),
    y=y_train_encoded
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(
        np.unique(y_train_encoded),
        class_weights_array
    )
}

print("\nClass weights:", class_weights)


# ==========================================
# 6. BUILD MLP MODEL
# ==========================================

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(284,)),

    tf.keras.layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ),

    tf.keras.layers.Dropout(0.4),

    tf.keras.layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ),

    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(
        num_classes,
        activation="softmax"
    )
])  

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 7. CALLBACKS
# ==========================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
]

# ==========================================
# 8. TRAIN
# ==========================================

history = model.fit(
    X_train_scaled,
    y_train_encoded,

    validation_data=(
        X_val_scaled,
        y_val_encoded
    ),

    epochs=100,
    batch_size=32,

    class_weight=class_weights,
    callbacks=callbacks,

    verbose=1
)


# ==========================================
# 9. VALIDATION PERFORMANCE
# ==========================================

val_probabilities = model.predict(
    X_val_scaled,
    verbose=0
)

val_predictions = np.argmax(
    val_probabilities,
    axis=1
)

val_accuracy = accuracy_score(
    y_val_encoded,
    val_predictions
)

val_macro_f1 = f1_score(
    y_val_encoded,
    val_predictions,
    average="macro"
)

print("\n===== VALIDATION RESULTS =====")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Macro F1: {val_macro_f1:.4f}")


# ==========================================
# 10. FINAL TEST EVALUATION
# ==========================================

test_probabilities = model.predict(
    X_test_scaled,
    verbose=0
)

test_predictions = np.argmax(
    test_probabilities,
    axis=1
)

test_accuracy = accuracy_score(
    y_test_encoded,
    test_predictions
)

test_macro_f1 = f1_score(
    y_test_encoded,
    test_predictions,
    average="macro"
)

test_weighted_f1 = f1_score(
    y_test_encoded,
    test_predictions,
    average="weighted"
)

print("\n" + "=" * 45)
print("FINAL TEST RESULTS")
print("=" * 45)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"Test Weighted F1: {test_weighted_f1:.4f}")

print("\nClassification Report:")

print(classification_report(
    y_test_encoded,
    test_predictions,
    target_names=label_encoder.classes_,
    zero_division=0
))

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test_encoded,
    test_predictions
)

print(cm)


# ==========================================
# 11. SAVE MODEL AND PREPROCESSORS
# ==========================================

Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)

model.save("models/cremad_mlp_v1.keras")

joblib.dump(
    scaler,
    "models/cremad_mlp_scaler_v1.pkl"
)

joblib.dump(
    label_encoder,
    "models/cremad_mlp_label_encoder_v1.pkl"
)

np.save(
    "outputs/cremad_mlp_test_predictions_v1.npy",
    test_predictions
)

np.save(
    "outputs/cremad_mlp_test_probabilities_v1.npy",
    test_probabilities
)

pd.DataFrame(history.history).to_csv(
    "outputs/cremad_mlp_training_history_v1.csv",
    index=False
)

print("\nMLP model saved.")
print("Scaler saved.")
print("Label encoder saved.")
print("Training history saved.")
print("\nCREMA-D MLP Experiment complete!")