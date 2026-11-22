import numpy as np
import pandas as pd

data = np.load("outputs/cremad_features_v1.npz")

X = data["X"]
y = data["y"]
speakers = data["speaker_ids"]
splits = data["splits"]
filenames = data["filenames"]

print("===== DATA SHAPES =====")
print("X:", X.shape)
print("y:", y.shape)
print("Speakers:", speakers.shape)
print("Splits:", splits.shape)
print("Filenames:", filenames.shape)

print("\n===== DATA VALIDITY =====")
print("NaN values:", np.isnan(X).sum())
print("Infinite values:", np.isinf(X).sum())
print("All finite:", np.isfinite(X).all())

print("\n===== SPLIT COUNTS =====")
for split in ["train", "validation", "test"]:
    mask = splits == split
    print(f"{split}: {mask.sum()} samples")

print("\n===== SPEAKER LEAKAGE CHECK =====")
train_speakers = set(speakers[splits == "train"])
val_speakers = set(speakers[splits == "validation"])
test_speakers = set(speakers[splits == "test"])

print("Train-Val overlap:", train_speakers & val_speakers)
print("Train-Test overlap:", train_speakers & test_speakers)
print("Val-Test overlap:", val_speakers & test_speakers)

print("\n===== EMOTION DISTRIBUTION =====")
for split in ["train", "validation", "test"]:
    print(f"\n{split.upper()}")
    mask = splits == split
    print(pd.Series(y[mask]).value_counts())

print("\n===== FEATURE STATISTICS =====")
print("Minimum:", X.min())
print("Maximum:", X.max())
print("Mean:", X.mean())
print("Standard deviation:", X.std())

print("\nVerification complete.")
