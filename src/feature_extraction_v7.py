
import os
import numpy as np
import librosa

# ==========================================
# 1. Paths and settings
# ==========================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

DATASET_DIR = os.path.join(
    PROJECT_ROOT, "dataset", "RAVDESS"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT, "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 22050
N_MFCC = 40

VALID_EMOTIONS = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}


# ==========================================
# 2. Feature statistics helper
# ==========================================

def summarize_features(features):
    """
    Calculate mean and standard deviation
    across time for each feature.
    """
    return np.concatenate([
        np.mean(features, axis=1),
        np.std(features, axis=1)
    ])


# ==========================================
# 3. Feature extraction function
# ==========================================

def extract_features(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    features = []

    # --------------------------------------
    # MFCC
    # --------------------------------------

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    features.extend(summarize_features(mfcc))

    # --------------------------------------
    # Delta MFCC
    # --------------------------------------

    delta_mfcc = librosa.feature.delta(mfcc)

    features.extend(
        summarize_features(delta_mfcc)
    )

    # --------------------------------------
    # Delta-Delta MFCC
    # --------------------------------------

    delta2_mfcc = librosa.feature.delta(
        mfcc,
        order=2
    )

    features.extend(
        summarize_features(delta2_mfcc)
    )

    # --------------------------------------
    # Chroma
    # --------------------------------------

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=sr
    )

    features.extend(
        summarize_features(chroma)
    )

    # --------------------------------------
    # Spectral Contrast
    # --------------------------------------

    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=sr
    )

    features.extend(
        summarize_features(spectral_contrast)
    )

    # --------------------------------------
    # Zero Crossing Rate
    # --------------------------------------

    zcr = librosa.feature.zero_crossing_rate(
        y=audio
    )

    features.extend(
        summarize_features(zcr)
    )

    # --------------------------------------
    # RMS Energy
    # --------------------------------------

    rms = librosa.feature.rms(
        y=audio
    )

    features.extend(
        summarize_features(rms)
    )

    # --------------------------------------
    # Pitch / Fundamental Frequency
    # --------------------------------------

    f0 = librosa.yin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr
    )

    # Remove unvoiced or invalid pitch values
    valid_f0 = f0[
        np.isfinite(f0) & (f0 > 0)
    ]

    if len(valid_f0) > 0:
        features.extend([
            np.mean(valid_f0),
            np.std(valid_f0)
        ])
    else:
        features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


# ==========================================
# 4. Process dataset
# ==========================================

X = []
y = []
actors = []
file_names = []

processed = 0
skipped = 0

print("Starting Experiment 7 feature extraction...\n")

for actor_folder in sorted(os.listdir(DATASET_DIR)):

    actor_path = os.path.join(
        DATASET_DIR,
        actor_folder
    )

    if not os.path.isdir(actor_path):
        continue

    for file_name in sorted(os.listdir(actor_path)):

        if not file_name.endswith(".wav"):
            continue

        parts = file_name.split("-")

        # RAVDESS filename structure
        emotion_code = parts[2]
        actor_id = int(parts[6].split(".")[0])

        # Keep only our 4 emotions
        if emotion_code not in VALID_EMOTIONS:
            continue

        file_path = os.path.join(
            actor_path,
            file_name
        )

        try:

            feature_vector = extract_features(
                file_path
            )

            X.append(feature_vector)
            y.append(VALID_EMOTIONS[emotion_code])
            actors.append(actor_id)
            file_names.append(file_name)

            processed += 1

            if processed % 50 == 0:
                print(
                    f"Processed {processed} files..."
                )

        except Exception as e:

            skipped += 1

            print(
                f"Skipped {file_name}: {e}"
            )


# ==========================================
# 5. Convert to arrays
# ==========================================

X = np.array(X, dtype=np.float32)
y = np.array(y)
actors = np.array(actors)
file_names = np.array(file_names)

print("\nFeature extraction complete!")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Actors shape:", actors.shape)
print("Skipped files:", skipped)


# ==========================================
# 6. Save extracted features
# ==========================================

output_path = os.path.join(
    OUTPUT_DIR,
    "features_v7.npz"
)

np.savez_compressed(
    output_path,
    X=X,
    y=y,
    actors=actors,
    file_names=file_names
)

print("\nSaved to:")
print(output_path)

print("\nFeature vector size:", X.shape[1])

