import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Load Experiment 3 features
data = np.load(
    os.path.join(OUTPUT_DIR, "mfcc_features_v2.npz")
)

X = data["X"]
y = data["y"]
actors = data["actors"].astype(int)

# Same actor split as Experiment 2
train_actors = np.arange(1, 17)
val_actors = np.arange(17, 21)
test_actors = np.arange(21, 25)

# Create masks
train_mask = np.isin(actors, train_actors)
val_mask = np.isin(actors, val_actors)
test_mask = np.isin(actors, test_actors)

# Split data
X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

actors_train = actors[train_mask]
actors_val = actors[val_mask]
actors_test = actors[test_mask]

# Display shapes
print("Experiment 3 Dataset Splitting Completed!\n")

print("Training set:", X_train.shape)
print("Validation set:", X_val.shape)
print("Testing set:", X_test.shape)

print("\nTraining emotions:", np.unique(y_train))
print("Validation emotions:", np.unique(y_val))
print("Testing emotions:", np.unique(y_test))

print("\nActors:")
print("Train:", np.unique(actors_train))
print("Validation:", np.unique(actors_val))
print("Test:", np.unique(actors_test))

# Save new splits
np.savez(
    os.path.join(OUTPUT_DIR, "train_data_v2.npz"),
    X=X_train,
    y=y_train,
    actors=actors_train
)

np.savez(
    os.path.join(OUTPUT_DIR, "validation_data_v2.npz"),
    X=X_val,
    y=y_val,
    actors=actors_val
)

np.savez(
    os.path.join(OUTPUT_DIR, "test_data_v2.npz"),
    X=X_test,
    y=y_test,
    actors=actors_test
)

print("\nExperiment 3 split files saved successfully!")