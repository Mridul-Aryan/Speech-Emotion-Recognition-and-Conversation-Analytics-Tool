import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Get project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Load dataset metadata
csv_path = os.path.join(PROJECT_ROOT, "outputs", "dataset_metadata.csv")
df = pd.read_csv(csv_path)

# Select one happy audio sample
sample = df[df["emotion"] == "happy"].iloc[0]

audio_path = sample["path"]

print("Selected audio:", sample["filename"])
print("Emotion:", sample["emotion"])

# Load audio
audio, sr = librosa.load(audio_path, sr=None)

# Calculate time duration
duration = len(audio) / sr

print("Sample rate:", sr, "Hz")
print("Duration:", round(duration, 2), "seconds")

# Create Mel spectrogram
mel_spectrogram = librosa.feature.melspectrogram(
    y=audio,
    sr=sr,
    n_mels=128,
    n_fft=2048,
    hop_length=512
)

# Convert to decibels
mel_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

# Plot waveform and spectrogram
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Waveform
librosa.display.waveshow(audio, sr=sr, ax=axes[0])
axes[0].set_title("Audio Waveform - Happy")
axes[0].set_xlabel("Time (seconds)")
axes[0].set_ylabel("Amplitude")

# Mel Spectrogram
img = librosa.display.specshow(
    mel_db,
    sr=sr,
    hop_length=512,
    x_axis="time",
    y_axis="mel",
    cmap="magma",
    ax=axes[1]
)

axes[1].set_title("Mel Spectrogram - Happy")
axes[1].set_xlabel("Time (seconds)")
axes[1].set_ylabel("Frequency (Hz)")

fig.colorbar(img, ax=axes[1], format="%+2.0f dB")
plt.tight_layout()
plt.show()