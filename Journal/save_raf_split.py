import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("Journal"))
from dataset.raf import data_split

LABEL_FILE = r".\RAF_train_label2.txt"
OUTPUT_DIR = r".\results_raf_20_epochs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(2)

labeled, unlabeled = data_split(
    LABEL_FILE,
    n_labeled=4000,
    n_labeled_fear=250
)

with open(LABEL_FILE) as f:
    lines = f.read().splitlines()

labeled_paths = [lines[i].split(" ")[0] for i in labeled]
unlabeled_paths = [lines[i].split(" ")[0] for i in unlabeled]

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "raf_split_seed2.npz"),
    labeled_indices=labeled,
    unlabeled_indices=unlabeled,
    labeled_paths=labeled_paths,
    unlabeled_paths=unlabeled_paths
)

print("SPLIT SAVED SUCCESSFULLY")
print("Labeled:", len(labeled))
print("Unlabeled:", len(unlabeled))

labels = np.array([int(line.split(" ")[1]) for line in lines])

for i, name in enumerate([
    "Surprise", "Fear", "Disgust", "Happiness",
    "Sadness", "Anger", "Neutral"
]):
    count = np.sum(labels[labeled] == i)
    print(f"{name:12s}: {count}")

print("Saved to:", os.path.join(OUTPUT_DIR, "raf_split_seed2.npz"))
