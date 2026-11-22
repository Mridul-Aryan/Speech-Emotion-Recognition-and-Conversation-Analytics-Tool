import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from tqdm import tqdm


# ==========================================
# CONFIGURATION
# ==========================================

SAMPLE_RATE = 22050
DURATION = 3.0

N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

MAX_SAMPLES = int(SAMPLE_RATE * DURATION)
MAX_FRAMES = 256

MANIFEST_PATH = "outputs/cremad_split_manifest.csv"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ==========================================
# LOAD MANIFEST
# ==========================================

df = pd.read_csv(MANIFEST_PATH)

print("Total samples:", len(df))
print(df["split"].value_counts())


# ==========================================
# MEL SPECTROGRAM FUNCTION
# ==========================================

def audio_to_logmel(file_path):

    # Load audio
    y, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Pad or truncate audio to 3 seconds
    if len(y) < MAX_SAMPLES:
        y = np.pad(
            y,
            (0, MAX_SAMPLES - len(y))
        )
    else:
        y = y[:MAX_SAMPLES]

    # Convert to Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=0,
        fmax=8000
    )

    # Convert power to decibels
    logmel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Ensure fixed time dimension
    if logmel.shape[1] < MAX_FRAMES:
        logmel = np.pad(
            logmel,
            ((0, 0), (0, MAX_FRAMES - logmel.shape[1]))
        )
    else:
        logmel = logmel[:, :MAX_FRAMES]

    return logmel.astype(np.float32)


# ==========================================
# PROCESS DATASET
# ==========================================

X_mel = []
labels = []
speakers = []
splits = []
filenames = []

failed_files = []

for _, row in tqdm(
    df.iterrows(),
    total=len(df),
    desc="Extracting Log-Mel"
):

    try:

        mel = audio_to_logmel(
            row["file_path"]
        )

        if mel.shape != (N_MELS, MAX_FRAMES):
            raise ValueError(
                f"Unexpected shape: {mel.shape}"
            )

        X_mel.append(mel)
        labels.append(row["emotion"])
        speakers.append(row["speaker_id"])
        splits.append(row["split"])
        filenames.append(row["filename"])

    except Exception as e:

        failed_files.append({
            "filename": row["filename"],
            "error": str(e)
        })


# ==========================================
# SAVE DATA
# ==========================================

X_mel = np.array(
    X_mel,
    dtype=np.float32
)

labels = np.array(labels)
speakers = np.array(speakers)
splits = np.array(splits)
filenames = np.array(filenames)


print("\n===== RESULTS =====")
print("Processed:", len(X_mel))
print("Failed:", len(failed_files))
print("Spectrogram shape:", X_mel.shape)

if len(failed_files) > 0:

    pd.DataFrame(failed_files).to_csv(
        OUTPUT_DIR / "cremad_mel_failed_files.csv",
        index=False
    )

    print("Failed files saved.")


np.savez_compressed(
    OUTPUT_DIR / "cremad_logmel_v1.npz",
    X=X_mel,
    y=labels,
    speaker_ids=speakers,
    splits=splits,
    filenames=filenames
)

print("\nSaved: outputs/cremad_logmel_v1.npz")
print("Log-Mel preparation complete!")