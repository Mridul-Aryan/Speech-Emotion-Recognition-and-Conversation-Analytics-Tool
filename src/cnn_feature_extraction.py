import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt

# ==========================================
# 1. Paths
# ==========================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

METADATA_PATH = os.path.join(OUTPUT_DIR, "dataset_metadata.csv")

# ==========================================
# 2. Settings
# ==========================================

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

TARGET_FRAMES = 160

# ==========================================
# 3. Load metadata
# ==========================================

df = pd.read_csv(METADATA_PATH)

X = []
y = []
actors = []

print("Starting CNN Mel Spectrogram Extraction...")
print(f"Total audio files: {len(df)}")

# ==========================================
# 4. Process each audio file
# ==========================================

for index, row in df.iterrows():

    audio_path = row["path"]
    emotion = row["emotion"]
    actor = row["actor"]

    # Load audio
    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )

    # Extract Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    # Convert to decibels
    mel_db = librosa.power_to_db(
        mel,
        ref=np.max,
        top_db=80
    )

    # Normalize from 0 to 1
    mel_normalized = (mel_db + 80) / 80
    mel_normalized = np.clip(mel_normalized, 0, 1)

    # Fix time dimension to TARGET_FRAMES
    current_frames = mel_normalized.shape[1]

    if current_frames < TARGET_FRAMES:

        pad_width = TARGET_FRAMES - current_frames

        mel_normalized = np.pad(
            mel_normalized,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:

        mel_normalized = mel_normalized[:, :TARGET_FRAMES]

    # Add to dataset
    X.append(mel_normalized)
    y.append(emotion)
    actors.append(actor)

    if (index + 1) % 50 == 0:
        print(f"Processed {index + 1}/{len(df)} files")

# ==========================================
# 5. Convert to arrays
# ==========================================

X = np.array(X)
y = np.array(y)
actors = np.array(actors)

# Add CNN channel dimension
X = X[..., np.newaxis]

print("\nCNN feature extraction completed!")
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)
print("Actors shape:", actors.shape)

# ==========================================
# 6. Save features
# ==========================================

np.savez(
    os.path.join(OUTPUT_DIR, "mel_spectrograms_cnn.npz"),
    X=X,
    y=y,
    actors=actors
)

print("CNN spectrogram features saved successfully!")

# ==========================================
# 7. Visualize sample
# ==========================================

sample_index = 0

plt.figure(figsize=(10, 5))

librosa.display.specshow(
    X[sample_index, :, :, 0],
    sr=SAMPLE_RATE,
    hop_length=HOP_LENGTH,
    x_axis="time",
    y_axis="mel"
)

plt.colorbar(format="%+2.0f")
plt.title(f"Mel Spectrogram - Emotion: {y[sample_index]}")
plt.tight_layout()
plt.show()