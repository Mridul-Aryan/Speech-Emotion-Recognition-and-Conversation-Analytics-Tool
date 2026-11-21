
import os
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load split data
# --------------------------------------------------

train_data = np.load(os.path.join(OUTPUT_DIR, "train_data.npz"))
val_data = np.load(os.path.join(OUTPUT_DIR, "validation_data.npz"))
test_data = np.load(os.path.join(OUTPUT_DIR, "test_data.npz"))

X_train = train_data["X"]
y_train = train_data["y"]

X_val = val_data["X"]
y_val = val_data["y"]

X_test = test_data["X"]
y_test = test_data["y"]

print("Data loaded successfully!")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

# --------------------------------------------------
# 3. Encode emotion labels
# --------------------------------------------------

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

print("\nEmotion classes:", label_encoder.classes_)

# --------------------------------------------------
# 4. Feature normalization
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# 5. Handle class imbalance
# --------------------------------------------------

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

# --------------------------------------------------
# 6. Build the MLP model
# --------------------------------------------------

model = Sequential([
    Input(shape=(80,)),

    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(4, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# 7. Training callbacks
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

# --------------------------------------------------
# 8. Train the model
# --------------------------------------------------

history = model.fit(
    X_train_scaled,
    y_train_encoded,
    validation_data=(X_val_scaled, y_val_encoded),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# --------------------------------------------------
# 9. Evaluate on test set
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    X_test_scaled,
    y_test_encoded,
    verbose=0
)

y_pred_prob = model.predict(X_test_scaled, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

print("\n" + "=" * 40)
print("FINAL TEST RESULTS")
print("=" * 40)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_test_encoded,
    y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred))

# --------------------------------------------------
# 10. Save model and preprocessing objects
# --------------------------------------------------

model.save(os.path.join(MODEL_DIR, "emotion_mlp.keras"))

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "scaler.pkl")
)

joblib.dump(
    label_encoder,
    os.path.join(MODEL_DIR, "label_encoder.pkl")
)

# Save training history
np.savez(
    os.path.join(OUTPUT_DIR, "training_history.npz"),
    loss=history.history["loss"],
    val_loss=history.history["val_loss"],
    accuracy=history.history["accuracy"],
    val_accuracy=history.history["val_accuracy"]
)

print("\nModel saved successfully!")
print("Saved in:", MODEL_DIR)
