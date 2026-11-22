import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

# ---------------- CONFIG ----------------
DATA_PATH = "outputs/cremad_logmel_v1.npz"
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = ["angry", "happy", "neutral", "sad"]
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- LOAD DATA ----------------
data = np.load(DATA_PATH, allow_pickle=True)

X = data["X"]          # (N, 64, 256)
y = data["y"]
splits = data["splits"]

print("Dataset shape:", X.shape)

# Encode emotions
class_to_idx = {emotion: i for i, emotion in enumerate(CLASSES)}
y_encoded = np.array([class_to_idx[label] for label in y])

train_idx = np.where(splits == "train")[0]
val_idx = np.where(splits == "validation")[0]
test_idx = np.where(splits == "test")[0]

# ---------------- NORMALIZATION ----------------
# Calculate normalization values using TRAIN data only
train_mean = X[train_idx].mean()
train_std = X[train_idx].std()

X = (X - train_mean) / (train_std + 1e-8)

print("Normalization mean:", train_mean)
print("Normalization std:", train_std)

# ---------------- DATASET ----------------
class EmotionDataset(Dataset):
    def __init__(self, X, y, indices):
        self.X = X
        self.y = y
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        # Add channel dimension: (64,256) -> (1,64,256)
        mel = torch.tensor(self.X[i], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.y[i], dtype=torch.long)

        return mel, label

train_dataset = EmotionDataset(X, y_encoded, train_idx)
val_dataset = EmotionDataset(X, y_encoded, val_idx)
test_dataset = EmotionDataset(X, y_encoded, test_idx)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# ---------------- CLASS WEIGHTS ----------------
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_encoded[train_idx]),
    y=y_encoded[train_idx]
)

class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
print("Class weights:", class_weights)

# ---------------- RESNET18 ----------------
model = resnet18(weights=None)

# Modify first layer to accept 1-channel input instead of 3
model.conv1 = nn.Conv2d(
    1, 64, kernel_size=7, stride=2, padding=3, bias=False
)

# Replace 1000-class ImageNet output with 4 emotion classes
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# ---------------- TRAINING ----------------
best_val_loss = float("inf")
patience_counter = 0

history = []

for epoch in range(EPOCHS):

    # ---- TRAIN ----
    model.train()

    train_loss_total = 0
    train_preds = []
    train_labels = []

    for mel, labels in train_loader:
        mel = mel.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(mel)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss_total += loss.item() * labels.size(0)

        preds = torch.argmax(outputs, dim=1)

        train_preds.extend(preds.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())

    train_loss = train_loss_total / len(train_dataset)
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average="macro")

    # ---- VALIDATION ----
    model.eval()

    val_loss_total = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for mel, labels in val_loader:
            mel = mel.to(device)
            labels = labels.to(device)

            outputs = model(mel)
            loss = criterion(outputs, labels)

            val_loss_total += loss.item() * labels.size(0)

            preds = torch.argmax(outputs, dim=1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_loss = val_loss_total / len(val_dataset)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="macro")

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
    f"Epoch {epoch+1:02d}/{EPOCHS} | "
    f"Train Loss: {train_loss:.4f} | "
    f"Train Acc: {train_acc:.4f} | "
    f"Train F1: {train_f1:.4f} | "
    f"Val Loss: {val_loss:.4f} | "
    f"Val Acc: {val_acc:.4f} | "
    f"Val F1: {val_f1:.4f} | "
    f"LR: {current_lr:.6f}"
)
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "train_macro_f1": train_f1,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
        "learning_rate": current_lr
    })

    # ---- EARLY STOPPING ----
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "train_mean": float(train_mean),
            "train_std": float(train_std),
            "best_val_loss": float(best_val_loss)
        }, MODEL_DIR / "cremad_resnet18_v1.pth")

        print("  -> Best model saved!")

    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

# Save training history
import pandas as pd

pd.DataFrame(history).to_csv(
    OUTPUT_DIR / "cremad_resnet18_history_v1.csv",
    index=False
)

# ---------------- TEST ----------------
print("\nLoading best model for testing...")

checkpoint = torch.load(
    MODEL_DIR / "cremad_resnet18_v1.pth",
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_preds = []
test_labels = []

with torch.no_grad():
    for mel, labels in test_loader:
        mel = mel.to(device)

        outputs = model(mel)
        preds = torch.argmax(outputs, dim=1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.numpy())

test_acc = accuracy_score(test_labels, test_preds)
test_macro_f1 = f1_score(test_labels, test_preds, average="macro")
test_weighted_f1 = f1_score(test_labels, test_preds, average="weighted")

print("\n========== FINAL TEST RESULTS ==========")
print(f"Accuracy:    {test_acc:.4f}")
print(f"Macro F1:   {test_macro_f1:.4f}")
print(f"Weighted F1: {test_weighted_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    test_labels,
    test_preds,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    digits=4
))

cm = confusion_matrix(
    test_labels,
    test_preds,
    labels=list(range(len(CLASSES)))
)

print("\nConfusion Matrix:")
print(cm)

np.save(
    OUTPUT_DIR / "cremad_resnet18_test_predictions_v1.npy",
    np.array(test_preds)
)

np.save(
    OUTPUT_DIR / "cremad_resnet18_test_labels_v1.npy",
    np.array(test_labels)
)

print("\nResNet18 training complete!")