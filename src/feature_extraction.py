import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

#project path 

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

CSV_PATH = os.path.join(
    PROJECT_ROOT, "outputs", "dataset_metadata.csv"
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset metadata
df = pd.read_csv(CSV_PATH)

features = []
labels = []
actors = []

# Extract MFCC features from every audio file
for index, row in df.iterrows():

    audio_path = row["path"]

    # Load audio at a consistent sample rate
    audio, sr = librosa.load(audio_path, sr=22050)

    # Extract 40 MFCC coefficients
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # Calculate mean and standard deviation
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Combine into an 80-dimensional feature vector
    combined_features = np.concatenate(
        [mfcc_mean, mfcc_std]
    )

    features.append(combined_features)
    labels.append(row["emotion"])
    actors.append(row["actor"])

    if (index + 1) % 50 == 0:
        print(f"Processed {index + 1}/{len(df)} files")

# Convert lists into NumPy arrays
X = np.array(features, dtype=np.float32)
y = np.array(labels)
actor_ids = np.array(actors)

print("\nMFCC extraction completed!")
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# Save extracted features
np.savez_compressed(
    os.path.join(OUTPUT_DIR, "mfcc_features.npz"),
    X=X,
    y=y,
    actors=actor_ids
)

print("Features saved successfully!")

# Visualize MFCCs from one audio sample
sample_audio, sr = librosa.load(
    df.iloc[0]["path"],
    sr=22050
)

sample_mfcc = librosa.feature.mfcc(
    y=sample_audio,
    sr=sr,
    n_mfcc=40
)

plt.figure(figsize=(12, 5))

librosa.display.specshow(
    sample_mfcc,
    sr=sr,
    x_axis="time",
    cmap="coolwarm"
)

plt.colorbar()
plt.title("MFCC Visualization")
plt.xlabel("Time")
plt.ylabel("MFCC Coefficients")

plt.tight_layout()
plt.show()