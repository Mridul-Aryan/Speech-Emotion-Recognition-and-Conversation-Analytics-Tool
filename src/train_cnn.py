import os
import random
import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    GlobalAveragePooling2D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

import joblib


# ==========================================
# 1. Reproducibility
# ==========================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ==========================================
# 2. Paths
# ==========================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# 3. Load CNN dataset
# ==========================================

train_data = np.load(
    os.path.join(OUTPUT_DIR, "cnn_train_data.npz")
)

val_data = np.load(
    os.path.join(OUTPUT_DIR, "cnn_validation_data.npz")
)

test_data = np.load(
    os.path.join(OUTPUT_DIR, "cnn_test_data.npz")
)

X_train = train_data["X"].astype("float32")
y_train = train_data["y"]

X_val = val_data["X"].astype("float32")
y_val = val_data["y"]

X_test = test_data["X"].astype("float32")
y_test = test_data["y"]

print("Dataset loaded successfully!")
print("X_train:", X_train.shape)
print("X_val:", X_val.shape)
print("X_test:", X_test.shape)


# ==========================================
# 4. Encode emotion labels
# ==========================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

print("\nEmotion classes:", label_encoder.classes_)


# ==========================================
# 5. Class weights
# ==========================================

classes = np.unique(y_train_encoded)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_encoded
)

class_weights = dict(zip(classes, weights))

print("\nClass weights:", class_weights)


# ==========================================
# 6. Build CNN model
# ==========================================

model = Sequential([
    Input(shape=(128, 160, 1)),

    # Convolution Block 1
    Conv2D(32, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Convolution Block 2
    Conv2D(64, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Convolution Block 3
    Conv2D(128, (3, 3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    # Classification section
    GlobalAveragePooling2D(),

    Dense(64, activation="relu"),
    Dropout(0.4),

    Dense(4, activation="softmax")
])


# ==========================================
# 7. Compile model
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 8. Callbacks
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=1e-6,
    verbose=1
)


# ==========================================
# 9. Train CNN
# ==========================================

history = model.fit(
    X_train,
    y_train_encoded,

    validation_data=(
        X_val,
        y_val_encoded
    ),

    epochs=100,
    batch_size=32,
    class_weight=class_weights,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ==========================================
# 10. Evaluate on test set
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test_encoded,
    verbose=0
)

y_pred_prob = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(y_pred_prob, axis=1)


print("\n" + "=" * 40)
print("FINAL CNN TEST RESULTS")
print("=" * 40)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")


# ==========================================
# 11. Classification Report
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ==========================================
# 12. Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test_encoded,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 13. Save Model
# ==========================================

model.save(
    os.path.join(MODEL_DIR, "emotion_cnn_v5.keras")
)

joblib.dump(
    label_encoder,
    os.path.join(MODEL_DIR, "label_encoder_cnn_v5.pkl")
)

print("\nCNN model saved successfully!")


# ==========================================
# 14. Save training history
# ==========================================

np.savez(
    os.path.join(OUTPUT_DIR, "training_history_cnn_v5.npz"),

    loss=history.history["loss"],
    val_loss=history.history["val_loss"],

    accuracy=history.history["accuracy"],
    val_accuracy=history.history["val_accuracy"]
)

print("CNN training history saved successfully!")