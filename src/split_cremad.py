from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split


# ==========================================
# PATHS
# ==========================================

DATASET_PATH = Path("dataset/CREMA-D/AudioWAV")
OUTPUT_PATH = Path("outputs")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ==========================================
# EMOTION MAPPING
# ==========================================

emotion_mapping = {
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad",
    "NEU": "neutral"
}

# Fixed class order for future model training
class_order = ["angry", "happy", "neutral", "sad"]


# ==========================================
# CREATE DATA MANIFEST
# ==========================================

records = []

for file in DATASET_PATH.glob("*.wav"):

    parts = file.stem.split("_")

    if len(parts) < 4:
        continue

    speaker_id = parts[0]
    emotion_code = parts[2]

    # Keep only our four target emotions
    if emotion_code not in emotion_mapping:
        continue

    records.append({
        "file_path": str(file),
        "filename": file.name,
        "speaker_id": speaker_id,
        "emotion_code": emotion_code,
        "emotion": emotion_mapping[emotion_code]
    })


df = pd.DataFrame(records)

print("=" * 50)
print("CREMA-D FILTERING")
print("=" * 50)

print("Total selected samples:", len(df))
print("Unique speakers:", df["speaker_id"].nunique())

print("\nEmotion distribution:")
print(df["emotion"].value_counts())


# ==========================================
# SPLIT BY SPEAKER
# ==========================================

speakers = sorted(df["speaker_id"].unique())

# First: 70% train, 30% remaining
train_speakers, remaining_speakers = train_test_split(
    speakers,
    test_size=0.30,
    random_state=42
)

# Second: divide remaining equally into validation and test
val_speakers, test_speakers = train_test_split(
    remaining_speakers,
    test_size=0.50,
    random_state=42
)


# Assign split labels
df["split"] = "unused"

df.loc[
    df["speaker_id"].isin(train_speakers),
    "split"
] = "train"

df.loc[
    df["speaker_id"].isin(val_speakers),
    "split"
] = "validation"

df.loc[
    df["speaker_id"].isin(test_speakers),
    "split"
] = "test"


# ==========================================
# VERIFY NO SPEAKER LEAKAGE
# ==========================================

train_ids = set(train_speakers)
val_ids = set(val_speakers)
test_ids = set(test_speakers)

assert train_ids.isdisjoint(val_ids)
assert train_ids.isdisjoint(test_ids)
assert val_ids.isdisjoint(test_ids)

assert len(train_ids) + len(val_ids) + len(test_ids) == len(speakers)


# ==========================================
# DISPLAY SPLIT STATISTICS
# ==========================================

print("\n" + "=" * 50)
print("SPLIT STATISTICS")
print("=" * 50)

for split in ["train", "validation", "test"]:

    split_df = df[df["split"] == split]

    print(f"\n{split.upper()}")
    print("-" * 30)

    print("Samples:", len(split_df))
    print("Speakers:", split_df["speaker_id"].nunique())

    print("\nEmotion distribution:")
    print(
        split_df["emotion"]
        .value_counts()
        .reindex(class_order, fill_value=0)
    )


# ==========================================
# SAVE MANIFEST
# ==========================================

manifest_path = OUTPUT_PATH / "cremad_split_manifest.csv"

df.to_csv(manifest_path, index=False)

print("\n" + "=" * 50)
print("SPEAKER LEAKAGE CHECK")
print("=" * 50)

print("Train speakers:", len(train_ids))
print("Validation speakers:", len(val_ids))
print("Test speakers:", len(test_ids))

print("Train-Val overlap:", train_ids.intersection(val_ids))
print("Train-Test overlap:", train_ids.intersection(test_ids))
print("Val-Test overlap:", val_ids.intersection(test_ids))

print("\nManifest saved to:", manifest_path.resolve())

print("\nDataset preparation complete!")
