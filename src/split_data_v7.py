
import os
import numpy as np

# ==========================================
# 1. Paths
# ==========================================

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

data = np.load(
    os.path.join(OUTPUT_DIR, "features_v7.npz")
)

X = data["X"]
y = data["y"]
actors = data["actors"]

# ==========================================
# 2. Actor-independent split
# ==========================================

train_actors = list(range(1, 17))
val_actors = list(range(17, 21))
test_actors = list(range(21, 25))

train_mask = np.isin(actors, train_actors)
val_mask = np.isin(actors, val_actors)
test_mask = np.isin(actors, test_actors)

X_train, y_train = X[train_mask], y[train_mask]
X_val, y_val = X[val_mask], y[val_mask]
X_test, y_test = X[test_mask], y[test_mask]

actors_train = actors[train_mask]
actors_val = actors[val_mask]
actors_test = actors[test_mask]

# ==========================================
# 3. Print dataset information
# ==========================================

print("=" * 40)
print("EXPERIMENT 7 DATA SPLIT")
print("=" * 40)

print("\nTraining:")
print("X:", X_train.shape)
print("y:", y_train.shape)

print("\nValidation:")
print("X:", X_val.shape)
print("y:", y_val.shape)

print("\nTesting:")
print("X:", X_test.shape)
print("y:", y_test.shape)

# ==========================================
# 4. Verify actors
# ==========================================

print("\nTrain actors:", np.unique(actors_train))
print("Validation actors:", np.unique(actors_val))
print("Test actors:", np.unique(actors_test))

# ==========================================
# 5. Verify emotion distribution
# ==========================================

for name, labels in [
    ("Train", y_train),
    ("Validation", y_val),
    ("Test", y_test)
]:
    print(f"\n{name} emotion distribution:")

    classes, counts = np.unique(
        labels,
        return_counts=True
    )

    for emotion, count in zip(classes, counts):
        print(f"{emotion}: {count}")

# ==========================================
# 6. Save splits
# ==========================================

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "train_data_v7.npz"),
    X=X_train,
    y=y_train,
    actors=actors_train
)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "validation_data_v7.npz"),
    X=X_val,
    y=y_val,
    actors=actors_val
)

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "test_data_v7.npz"),
    X=X_test,
    y=y_test,
    actors=actors_test
)

print("\nExperiment 7 splits saved successfully!")
