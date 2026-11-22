from pathlib import Path
from pyexpat import features
import numpy as np
import pandas as pd
import librosa


# ==========================================
# CONFIGURATION
# ==========================================

MANIFEST_PATH = Path("outputs/cremad_split_manifest.csv")
OUTPUT_PATH = Path("outputs")

SAMPLE_RATE = 22050
N_MFCC = 40

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ==========================================
# FEATURE HELPER
# ==========================================

def mean_std(feature):
    """
    Calculate mean and standard deviation
    across time frames.
    """
    return np.concatenate([
        np.mean(feature, axis=1),
        np.std(feature, axis=1)
    ])


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(file_path):

    # Load audio as mono and resample
    y, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    # Prevent processing empty audio
    if len(y) == 0:
        raise ValueError("Empty audio file")

    features = []

    # --------------------------------------
    # 1. MFCC
    # --------------------------------------
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC
    )

    features.extend(mean_std(mfcc))

    # --------------------------------------
    # 2. Delta MFCC
    # --------------------------------------
    delta = librosa.feature.delta(
        mfcc,
        order=1
    )

    features.extend(mean_std(delta))

    # --------------------------------------
    # 3. Delta-Delta MFCC
    # --------------------------------------
    delta_delta = librosa.feature.delta(
        mfcc,
        order=2
    )

    features.extend(mean_std(delta_delta))

    # --------------------------------------
    # 4. Chroma
    # --------------------------------------
    chroma = librosa.feature.chroma_stft(
        y=y,
        sr=sr
    )

    features.extend(mean_std(chroma))

    # --------------------------------------
    # 5. Spectral Contrast
    # --------------------------------------
    spectral_contrast = librosa.feature.spectral_contrast(
    y=y,
    sr=sr,
    fmin=80.0,
    n_bands=6
)

    features.extend(mean_std(spectral_contrast))

    # --------------------------------------
    # 6. Zero Crossing Rate
    # --------------------------------------
    zcr = librosa.feature.zero_crossing_rate(y)

    features.extend(mean_std(zcr))

    # --------------------------------------
    # 7. RMS Energy
    # --------------------------------------
    rms = librosa.feature.rms(y=y)

    features.extend(mean_std(rms))

    # --------------------------------------
    # 8. Pitch using YIN
    # --------------------------------------
    pitch = librosa.yin(
        y,
        fmin=50,
        fmax=500,
        sr=sr
    )

    pitch = pitch.reshape(1, -1)

    features.extend(mean_std(pitch))

    return np.asarray(features, dtype=np.float32)


# ==========================================
# LOAD MANIFEST
# ==========================================

df = pd.read_csv(MANIFEST_PATH)

print("=" * 60)
print("CREMA-D FEATURE EXTRACTION")
print("=" * 60)

print("Total files to process:", len(df))


# ==========================================
# EXTRACT FEATURES
# ==========================================

X = []
y = []
speaker_ids = []
splits = []
filenames = []

failed_files = []

for i, row in df.iterrows():

    file_path = Path(row["file_path"])

    try:

        feature_vector = extract_features(file_path)

        # Check dimensions
        if feature_vector.shape[0] != 284:
            raise ValueError(
                f"Unexpected feature dimension: "
                f"{feature_vector.shape[0]}"
            )

        # Check for NaN or infinite values
        if not np.isfinite(feature_vector).all():
            raise ValueError("NaN or infinite features detected")

        X.append(feature_vector)

        y.append(row["emotion"])
        speaker_ids.append(row["speaker_id"])
        splits.append(row["split"])
        filenames.append(row["filename"])

    except Exception as e:

        failed_files.append({
            "filename": row["filename"],
            "error": str(e)
        })

    # Progress
    if (i + 1) % 250 == 0:
        print(f"Processed {i + 1}/{len(df)} files")


# ==========================================
# CONVERT TO ARRAYS
# ==========================================

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)
speaker_ids = np.asarray(speaker_ids)
splits = np.asarray(splits)
filenames = np.asarray(filenames)


# ==========================================
# SAVE FEATURES
# ==========================================

output_file = OUTPUT_PATH / "cremad_features_v1.npz"

np.savez_compressed(
    output_file,
    X=X,
    y=y,
    speaker_ids=speaker_ids,
    splits=splits,
    filenames=filenames
)


# ==========================================
# REPORT
# ==========================================

print("\n" + "=" * 60)
print("FEATURE EXTRACTION RESULTS")
print("=" * 60)

print("Successfully processed:", len(X))
print("Failed files:", len(failed_files))

print("Feature shape:", X.shape)
print("Labels shape:", y.shape)
print("Speakers shape:", speaker_ids.shape)
print("Splits shape:", splits.shape)

if len(X) > 0:

    print("\nOverall emotion distribution:")
    print(pd.Series(y).value_counts())

    print("\nSplit distribution:")
    print(pd.Series(splits).value_counts())

    print("\nEmotion distribution by split:")
    print(
        pd.crosstab(
            splits,
            y,
            rownames=["Split"],
            colnames=["Emotion"]
        )
    )

if failed_files:

    print("\nFirst failed files:")

    for item in failed_files[:10]:
        print(item["filename"], "->", item["error"])

    # Save failed file report
    failed_df = pd.DataFrame(failed_files)

    failed_df.to_csv(
        OUTPUT_PATH / "cremad_failed_files.csv",
        index=False
    )

print("\nSaved to:", output_file.resolve())
print("\nFeature extraction complete!")