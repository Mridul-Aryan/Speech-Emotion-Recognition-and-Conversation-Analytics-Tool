from pathlib import Path
from collections import Counter

# Dataset path
DATASET_PATH = Path("dataset/CREMA-D/AudioWAV")

# Check if folder exists
if not DATASET_PATH.exists():
    print("❌ Dataset folder not found!")
    print(f"Expected location: {DATASET_PATH.resolve()}")
    exit()

# Get all WAV files
audio_files = list(DATASET_PATH.glob("*.wav"))

print("=" * 50)
print("CREMA-D DATASET INSPECTION")
print("=" * 50)

print(f"\nTotal audio files found: {len(audio_files)}")

# Counters
emotion_counts = Counter()
speaker_counts = Counter()
invalid_files = []

# Emotion mapping
emotion_mapping = {
    "ANG": "Angry",
    "HAP": "Happy",
    "SAD": "Sad",
    "NEU": "Neutral",
    "DIS": "Disgust",
    "FEA": "Fear"
}

for file in audio_files:

    # CREMA-D filename format:
    # 1001_DFA_ANG_XX.wav
    parts = file.stem.split("_")

    if len(parts) < 4:
        invalid_files.append(file.name)
        continue

    speaker_id = parts[0]
    emotion_code = parts[2]

    speaker_counts[speaker_id] += 1
    emotion_counts[emotion_code] += 1

# Display emotion distribution
print("\n--- Emotion Distribution ---")

for emotion_code, count in sorted(emotion_counts.items()):
    emotion_name = emotion_mapping.get(emotion_code, "Unknown")
    print(f"{emotion_name:12} ({emotion_code}): {count}")

# Display speaker statistics
print("\n--- Speaker Statistics ---")
print(f"Unique speakers: {len(speaker_counts)}")

print("\nFiles per speaker:")
for speaker, count in sorted(speaker_counts.items()):
    print(f"Speaker {speaker}: {count}")

# Invalid filename check
print("\n--- Filename Validation ---")
print(f"Invalid filenames: {len(invalid_files)}")

if invalid_files:
    print("\nInvalid file examples:")
    for filename in invalid_files[:10]:
        print(filename)

# Check our selected four emotions
selected_emotions = ["ANG", "HAP", "SAD", "NEU"]

selected_total = sum(
    emotion_counts[emotion]
    for emotion in selected_emotions
)

print("\n--- Our Selected 4 Emotions ---")
print(f"Total selected samples: {selected_total}")

for emotion in selected_emotions:
    print(
        f"{emotion_mapping[emotion]:12}: "
        f"{emotion_counts[emotion]}"
    )

print("\nInspection complete.")