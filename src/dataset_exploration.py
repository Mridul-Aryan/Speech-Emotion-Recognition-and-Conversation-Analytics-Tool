import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset directory
DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "dataset",
    "RAVDESS"
)

# Emotion labels from RAVDESS
emotion_mapping = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry"
}

#Load and label the files

data = []

for actor_folder in os.listdir(DATASET_PATH):

    actor_path = os.path.join(DATASET_PATH, actor_folder)

    if not os.path.isdir(actor_path):
        continue

    for filename in os.listdir(actor_path):

        if not filename.endswith(".wav"):
            continue

        parts = filename.split("-")

        modality = parts[0]
        vocal_channel = parts[1]
        emotion_code = parts[2]
        actor_id = parts[6].split(".")[0]

        # Audio-only speech and selected emotions
        if modality == "03" and vocal_channel == "01" and emotion_code in emotion_mapping:

            data.append({
                "filename": filename,
                "path": os.path.join(actor_path, filename),
                "emotion": emotion_mapping[emotion_code],
                "actor": int(actor_id)
            })

df = pd.DataFrame(data)

print("Dataset successfully loaded!")
print("Total selected files:", len(df))

print("\nFirst 5 rows:")
print(df.head())

print("\nEmotion counts:")
print(df["emotion"].value_counts())


# Visualize the distribution of emotions

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="emotion",
    order=["neutral", "happy", "sad", "angry"]
)

plt.title("RAVDESS Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Number of Audio Samples")

plt.tight_layout()
plt.show()