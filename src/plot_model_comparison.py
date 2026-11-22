import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

Path("outputs/figures").mkdir(parents=True, exist_ok=True)

models = ["SVM", "MLP", "ResNet18"]

accuracy = [65.48, 65.48, 67.72]
macro_f1 = [65.17, 64.98, 65.68]
weighted_f1 = [65.51, 65.37, 66.13]

x = np.arange(len(models))
width = 0.25

plt.figure(figsize=(10, 6))

plt.bar(x - width, accuracy, width, label="Accuracy")
plt.bar(x, macro_f1, width, label="Macro F1")
plt.bar(x + width, weighted_f1, width, label="Weighted F1")

plt.ylabel("Score (%)")
plt.xlabel("Model")
plt.title("CREMA-D Model Performance Comparison")
plt.xticks(x, models)
plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("outputs/figures/model_comparison_cremad.png", dpi=300)
plt.show()

print("Chart saved: outputs/figures/model_comparison_cremad.png")