
import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    SpatialDropout2D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
# 3. Load existing CNN dataset
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
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ==========================================
# 4. Encode labels
# ==========================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

print("\nEmotion classes:", label_encoder.classes_)


# ==========================================
# 5. Calculate class weights
# ==========================================

classes = np.unique(y_train_encoded)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train_encoded
)

class_weights = {
    int(cls): float(weight)
    for cls, weight in zip(classes, weights)
}

print("\nClass weights:", class_weights)


# ==========================================
# 6. Build smaller CNN (No Batch Normalization)
# ==========================================

model = Sequential([
    Input(shape=(128, 160, 1)),

    # Block 1
    Conv2D(
        16,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(0.001)
    ),
    MaxPooling2D((2, 2)),
    SpatialDropout2D(0.15),

    # Block 2
    Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(0.001)
    ),
    MaxPooling2D((2, 2)),
    SpatialDropout2D(0.20),

    # Block 3
    Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same",
        kernel_regularizer=l2(0.001)
    ),
    MaxPooling2D((2, 2)),
    SpatialDropout2D(0.25),

    # Feature aggregation
    GlobalAveragePooling2D(),

    # Classifier
    Dense(
        32,
        activation="relu",
        kernel_regularizer=l2(0.001)
    ),

    Dropout(0.5),

    Dense(4, activation="softmax")
])


# ==========================================
# 7. Compile
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ==========================================
# 8. Callbacks
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)


# ==========================================
# 9. Train
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
# 10. Evaluate model in inference mode
# ==========================================

print("\nEvaluating datasets in inference mode...")

train_loss, train_accuracy = model.evaluate(
    X_train,
    y_train_encoded,
    verbose=0
)

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val_encoded,
    verbose=0
)

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test_encoded,
    verbose=0
)


print("\n" + "=" * 50)
print("EXPERIMENT 6 - FINAL RESULTS")
print("=" * 50)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print(f"\nTrain Loss: {train_loss:.4f}")
print(f"Validation Loss: {val_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")


# ==========================================
# 11. Test predictions
# ==========================================

y_pred_prob = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    y_pred_prob,
    axis=1
)


# ==========================================
# 12. Classification report
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
# 13. Confusion matrix
# ==========================================

cm = confusion_matrix(
    y_test_encoded,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


# ==========================================
# 14. Prediction distribution
# ==========================================

print("\nPredicted class distribution:")

predicted_classes, predicted_counts = np.unique(
    y_pred,
    return_counts=True
)

for cls, count in zip(predicted_classes, predicted_counts):
    print(
        f"{label_encoder.classes_[cls]}: {count}"
    )


# ==========================================
# 15. Save model
# ==========================================

model.save(
    os.path.join(
        MODEL_DIR,
        "emotion_cnn_v6.keras"
    )
)

joblib.dump(
    label_encoder,
    os.path.join(
        MODEL_DIR,
        "label_encoder_cnn_v6.pkl"
    )
)

print("\nExperiment 6 model saved!")


# ==========================================
# 16. Save training history
# ==========================================

np.savez(
    os.path.join(
        OUTPUT_DIR,
        "training_history_cnn_v6.npz"
    ),

    loss=history.history["loss"],
    val_loss=history.history["val_loss"],

    accuracy=history.history["accuracy"],
    val_accuracy=history.history["val_accuracy"],

    train_accuracy=train_accuracy,
    final_val_accuracy=val_accuracy,
    test_accuracy=test_accuracy,

    train_loss=train_loss,
    val_loss_final=val_loss,
    test_loss=test_loss
)

print("Training history saved!")
print("\nExperiment 6 completed successfully!")
