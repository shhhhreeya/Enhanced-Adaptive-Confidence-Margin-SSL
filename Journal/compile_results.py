import csv
import json
import numpy as np
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

UNSUPERVISED_DIR = ROOT / "results_unsupervised_raf"
OUTPUT_DIR = ROOT / "results_comparison"

OUTPUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# LOAD UNSUPERVISED LINEAR PROBE RESULTS
# --------------------------------------------------

linear_data = np.load(
    UNSUPERVISED_DIR / "linear_probe_results.npz"
)

linear_probe = {
    "accuracy": float(linear_data["accuracy"]),
    "macro_f1": float(linear_data["macro_f1"]),
    "confusion_matrix": linear_data["confusion_matrix"].tolist(),
    "number_of_test_samples": int(len(linear_data["labels"]))
}

# --------------------------------------------------
# LOAD UNSUPERVISED CLUSTERING RESULTS
# --------------------------------------------------

clustering_data = np.load(
    UNSUPERVISED_DIR / "unsupervised_clustering_results.npz",
    allow_pickle=True
)

clustering_results = []

for result in clustering_data["results"]:
    clustering_results.append({
        "checkpoint": str(result["checkpoint"]),
        "epoch": int(result["epoch"]),
        "ARI": float(result["ari"]),
        "NMI": float(result["nmi"]),
        "Hungarian_accuracy": float(result["hungarian_accuracy"]),
        "cluster_counts": result["cluster_counts"].tolist()
    })

# --------------------------------------------------
# SUPERVISED AND SEMI-SUPERVISED RESULTS
# --------------------------------------------------

supervised = {
    "accuracy": 0.8191003911342895,
    "macro_f1": 0.7513,
    "best_epoch": 8,
    "evaluation": "End-to-end fine-tuned classifier"
}

eacm = {
    "accuracy": 0.84452412,
    "macro_f1": 0.7774,
    "best_epoch": 9,
    "evaluation": "End-to-end semi-supervised EACM classifier",
    "pseudo_label_acceptance_at_epoch_20": 0.4433
}

# --------------------------------------------------
# MAIN COMPARISON
# --------------------------------------------------

comparison = [
    {
        "approach": "Supervised",
        "accuracy": supervised["accuracy"],
        "macro_f1": supervised["macro_f1"],
        "evaluation": supervised["evaluation"]
    },
    {
        "approach": "Unsupervised",
        "accuracy": linear_probe["accuracy"],
        "macro_f1": linear_probe["macro_f1"],
        "evaluation": "Frozen-backbone linear probe"
    },
    {
        "approach": "Semi-supervised EACM",
        "accuracy": eacm["accuracy"],
        "macro_f1": eacm["macro_f1"],
        "evaluation": eacm["evaluation"]
    }
]

accuracy_improvement = (
    eacm["accuracy"] - supervised["accuracy"]
) * 100

f1_improvement = (
    eacm["macro_f1"] - supervised["macro_f1"]
) * 100

# --------------------------------------------------
# COMPLETE MASTER SUMMARY
# --------------------------------------------------

master_results = {
    "project": "Comparison of Supervised, Unsupervised and Semi-Supervised Learning for Facial Emotion Recognition",
    "dataset": {
        "name": "RAF-DB",
        "training_images": 12271,
        "testing_images": 3068,
        "number_of_classes": 7,
        "labeled_training_images_for_supervised_and_eacm": 4000,
        "unlabeled_training_images_for_eacm": 8271,
        "split_seed": 2
    },
    "results": {
        "supervised": supervised,
        "unsupervised": {
            **linear_probe,
            "clustering": clustering_results
        },
        "semi_supervised_eacm": eacm
    },
    "comparison": comparison,
    "eacm_improvement_over_supervised": {
        "accuracy_percentage_points": round(accuracy_improvement, 2),
        "macro_f1_percentage_points": round(f1_improvement, 2)
    },
    "methodology_and_limitations": [
        "Results are from a single random seed.",
        "Supervised and EACM checkpoints were selected using test accuracy, introducing test-set leakage.",
        "The unsupervised model was evaluated using a frozen-backbone linear probe, whereas supervised and EACM models were fine-tuned end-to-end.",
        "The linear probe used 4000 labeled training images and a fixed 100-epoch training schedule without a validation set.",
        "Unsupervised clustering metrics are not directly comparable to classifier accuracy.",
        "The RAF-DB dataset was obtained from a third-party Kaggle distribution; its provenance and licensing should be disclosed."
    ]
}

# --------------------------------------------------
# SAVE MASTER JSON
# --------------------------------------------------

json_path = OUTPUT_DIR / "master_results.json"

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(master_results, f, indent=4)

# --------------------------------------------------
# SAVE COMPARISON CSV
# --------------------------------------------------

csv_path = OUTPUT_DIR / "comparison_metrics.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["approach", "accuracy", "macro_f1", "evaluation"]
    )
    writer.writeheader()
    writer.writerows(comparison)

# --------------------------------------------------
# SAVE CLUSTERING CSV
# --------------------------------------------------

cluster_path = OUTPUT_DIR / "unsupervised_clustering.csv"

with open(cluster_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "checkpoint",
            "epoch",
            "ARI",
            "NMI",
            "Hungarian_accuracy"
        ]
    )
    writer.writeheader()

    for result in clustering_results:
        writer.writerow({
            key: result[key]
            for key in writer.fieldnames
        })

# --------------------------------------------------
# DISPLAY SUMMARY
# --------------------------------------------------

print("\n" + "=" * 65)
print("           EXPERIMENT RESULTS CONSOLIDATED")
print("=" * 65)

print(f"\n{'Approach':<25} {'Accuracy':>12} {'Macro F1':>12}")
print("-" * 52)

for row in comparison:
    print(
        f"{row['approach']:<25}"
        f"{row['accuracy'] * 100:>11.2f}%"
        f"{row['macro_f1'] * 100:>11.2f}%"
    )

print("\nEACM improvement over supervised:")
print(f"Accuracy: +{accuracy_improvement:.2f} percentage points")
print(f"Macro F1: +{f1_improvement:.2f} percentage points")

print("\nFiles created:")
print(json_path)
print(csv_path)
print(cluster_path)

print("\nAll results consolidated successfully!")
