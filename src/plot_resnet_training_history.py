import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("outputs/figures").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("outputs/cremad_resnet18_history_v1.csv")

# ---------------- LOSS CURVE ----------------
plt.figure(figsize=(9, 5))

plt.plot(df["epoch"], df["train_loss"], marker="o", label="Training Loss")
plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("ResNet18 Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/figures/resnet18_loss_curve.png", dpi=300)
plt.show()

# ---------------- ACCURACY CURVE ----------------
plt.figure(figsize=(9, 5))

plt.plot(df["epoch"], df["train_accuracy"], marker="o", label="Training Accuracy")
plt.plot(df["epoch"], df["val_accuracy"], marker="o", label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("ResNet18 Training and Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("outputs/figures/resnet18_accuracy_curve.png", dpi=300)
plt.show()

print("Training history graphs saved!")