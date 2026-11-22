import os
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

METADATA_PATH = os.path.join(OUTPUT_DIR, "dataset_metadata.csv")

# Load metadata
df = pd.read_csv(METADATA_PATH)

X = []
y = []
actors = []

# Feature extraction settings
SAMPLE_RATE = 22050
N_MFCC = 40
N_SEGMENTS = 3

print("Starting improved MFCC extraction...")

for index, row in df.iterrows():

    audio_path = row["path"]
    emotion = row["emotion"]
    actor = row["actor"]

    # Load audio
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Divide MFCC time frames into 3 segments
    frame_indices = np.array_split(
        np.arange(mfcc.shape[1]),
        N_SEGMENTS
    )

    clip_features = []

    for segment_indices in frame_indices:

        segment_mfcc = mfcc[:, segment_indices]

        # Mean of each MFCC
        segment_mean = np.mean(
            segment_mfcc,
            axis=1
        )

        # Standard deviation of each MFCC
        segment_std = np.std(
            segment_mfcc,
            axis=1
        )

        # Combine mean and standard deviation
        segment_features = np.concatenate([
            segment_mean,
            segment_std
        ])

        clip_features.extend(segment_features)

    X.append(clip_features)
    y.append(emotion)
    actors.append(actor)

    if (index + 1) % 50 == 0:
        print(f"Processed {index + 1}/{len(df)} audio files")

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)
actors = np.array(actors)

print("\nImproved feature extraction completed!")
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# Save features
np.savez(
    os.path.join(OUTPUT_DIR, "mfcc_features_v2.npz"),
    X=X,
    y=y,
    actors=actors
)

print("Features saved successfully!")