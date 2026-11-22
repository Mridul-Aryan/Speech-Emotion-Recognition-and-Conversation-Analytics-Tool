import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Emotion order used by LabelEncoder
labels = ["angry", "happy", "neutral", "sad"]

# Experiment 3 confusion matrix
cm = np.array([
    [27,  4, 1, 0],
    [11, 16, 3, 2],
    [ 8,  0, 4, 4],
    [ 7,  1, 2, 22]
])

# ==========================================
# 1. Confusion Matrix
# ==========================================

plt.figure(figsize=(7, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Experiment 3 - Confusion Matrix")
plt.xlabel("Predicted Emotion")
plt.ylabel("True Emotion")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrix_v3.png"),
    dpi=300
)

plt.show()

# ==========================================
# 2. Training & Validation Accuracy
# ==========================================

history_data = np.load(
    os.path.join(OUTPUT_DIR, "training_history_v3.npz")
)

plt.figure(figsize=(8, 5))

plt.plot(
    history_data["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history_data["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Experiment 3 - Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "accuracy_curve_v3.png"),
    dpi=300
)

plt.show()

# ==========================================
# 3. Training & Validation Loss
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    history_data["loss"],
    label="Training Loss"
)

plt.plot(
    history_data["val_loss"],
    label="Validation Loss"
)

plt.title("Experiment 3 - Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "loss_curve_v3.png"),
    dpi=300
)

plt.show()

print("Experiment 3 visualizations generated!")
print("Saved in outputs folder:")
print("- confusion_matrix_v3.png")
print("- accuracy_curve_v3.png")
print("- loss_curve_v3.png")