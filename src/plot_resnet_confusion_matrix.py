import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from pathlib import Path

Path("outputs/figures").mkdir(parents=True, exist_ok=True)

CLASSES = ["angry", "happy", "neutral", "sad"]

y_true = np.load("outputs/cremad_resnet18_test_labels_v1.npy")
y_pred = np.load("outputs/cremad_resnet18_test_predictions_v1.npy")

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

fig, ax = plt.subplots(figsize=(8, 6))

im = ax.imshow(cm, interpolation="nearest")

ax.set_title("ResNet18 Confusion Matrix - CREMA-D")
ax.set_xlabel("Predicted Emotion")
ax.set_ylabel("Actual Emotion")

ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))

ax.set_xticklabels(CLASSES)
ax.set_yticklabels(CLASSES)

# Annotate each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=12
        )

fig.colorbar(im, ax=ax, label="Number of Samples")
plt.tight_layout()

plt.savefig(
    "outputs/figures/resnet18_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Confusion matrix saved!")
print(cm)