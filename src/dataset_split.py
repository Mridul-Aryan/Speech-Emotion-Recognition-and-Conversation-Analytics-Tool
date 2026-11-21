import os
import numpy as np

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# Load extracted features
data_path = os.path.join(PROJECT_ROOT, "outputs", "mfcc_features.npz")
data = np.load(data_path)

X = data["X"]
y = data["y"]
actors = data["actors"]

# Define actor splits
train_actors = list(range(1, 17))   # Actors 1-16
val_actors = list(range(17, 21))    # Actors 17-20
test_actors = list(range(21, 25))   # Actors 21-24

# Create masks
train_mask = np.isin(actors, train_actors)
val_mask = np.isin(actors, val_actors)
test_mask = np.isin(actors, test_actors)

# Split data
X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

# Save the split data
output_path = os.path.join(PROJECT_ROOT, "outputs")

np.savez_compressed(
    os.path.join(output_path, "train_data.npz"),
    X=X_train, y=y_train
)

np.savez_compressed(
    os.path.join(output_path, "validation_data.npz"),
    X=X_val, y=y_val
)

np.savez_compressed(
    os.path.join(output_path, "test_data.npz"),
    X=X_test, y=y_test
)

# Display results
print("Dataset splitting completed!\n")

print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Testing set:", X_test.shape)

print("\nTraining emotions:", np.unique(y_train))
print("Validation emotions:", np.unique(y_val))
print("Testing emotions:", np.unique(y_test))

print("\nActors in each set:")
print("Train:", np.unique(actors[train_mask]))
print("Validation:", np.unique(actors[val_mask]))
print("Test:", np.unique(actors[test_mask]))



'''
Suppose Actor 1 records 60 audio files.

If you randomly split the recordings:
- 45 recordings from Actor 1 could enter training.
- 10 recordings from Actor 1 could enter validation.
- 5 recordings from Actor 1 could enter testing.
The model has already heard Actor 1's voice during training.

It might learn characteristics of the speaker, such as:
- Pitch
- Voice texture
- Accent
- Speaking speed
- Vocal characteristics
Instead of genuinely recognizing emotion, it could partly learn to recognize the speaker.

This is called data leakage in this context—information related to the test samples, particularly speaker identity, 
leaks into the training process and makes evaluation less independent.
'''