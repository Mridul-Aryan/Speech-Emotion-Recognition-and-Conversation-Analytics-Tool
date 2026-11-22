import numpy as np

data = np.load("outputs/cremad_logmel_v1.npz", allow_pickle=True)

splits = data["splits"]

print("Unique split values:", np.unique(splits))
print("Split counts:")

for split in np.unique(splits):
    print(repr(split), np.sum(splits == split))