import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

# Load model and encoder
model = load_model(
    os.path.join(MODEL_DIR, "emotion_cnn_v5.keras")
)

label_encoder = joblib.load(
    os.path.join(MODEL_DIR, "label_encoder_cnn_v5.pkl")
)

# Load all splits
train = np.load(os.path.join(OUTPUT_DIR, "cnn_train_data.npz"))
val = np.load(os.path.join(OUTPUT_DIR, "cnn_validation_data.npz"))
test = np.load(os.path.join(OUTPUT_DIR, "cnn_test_data.npz"))

datasets = {
    "Train": train["X"],
    "Validation": val["X"],
    "Test": test["X"]
}

# Diagnose predictions
for name, X in datasets.items():

    predictions = model.predict(X, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    predicted_labels = label_encoder.inverse_transform(
        predicted_classes
    )

    unique, counts = np.unique(
        predicted_labels,
        return_counts=True
    )

    print("Mean prediction probabilities:")
    print(np.mean(predictions, axis=0))

    print(f"\n{'=' * 35}")
    print(f"{name} Prediction Distribution")
    print("=" * 35)

    for emotion, count in zip(unique, counts):
        percentage = count / len(X) * 100

        print(
            f"{emotion:10s}: {count:3d} "
            f"({percentage:.2f}%)"
        )

    print("Total predictions:", len(predicted_labels))

    print("\nActual label distribution in each dataset:")

for name, dataset in [
    ("Train", train),
    ("Validation", val),
    ("Test", test)
]:
    labels, counts = np.unique(dataset["y"], return_counts=True)

    print(f"\n{name}:")
    for label, count in zip(labels, counts):
        print(f"{label}: {count}")