import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "results_comparison"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSES = [
    "Surprise", "Fear", "Disgust",
    "Happiness", "Sadness", "Anger", "Neutral"
]

# --------------------------------------------------
# LOAD CONSOLIDATED RESULTS
# --------------------------------------------------

with open(OUTPUT_DIR / "master_results.json", "r") as f:
    results = json.load(f)

comparison = results["comparison"]

# --------------------------------------------------
# CONFUSION MATRICES
# Rows = Actual classes, Columns = Predicted classes
# --------------------------------------------------

supervised_cm = np.array([
    [236, 14, 18, 8, 7, 7, 39],
    [2, 45, 2, 7, 6, 5, 7],
    [0, 1, 123, 9, 10, 9, 8],
    [11, 2, 31, 1051, 26, 17, 47],
    [1, 3, 35, 14, 362, 12, 51],
    [1, 3, 15, 4, 3, 130, 6],
    [6, 1, 45, 23, 34, 5, 566]
])

eacm_cm = np.array([
    [281, 15, 7, 3, 4, 6, 13],
    [11, 47, 2, 1, 6, 4, 3],
    [3, 1, 118, 7, 8, 12, 11],
    [24, 6, 33, 1045, 13, 12, 52],
    [5, 5, 21, 6, 393, 10, 38],
    [3, 5, 7, 2, 2, 138, 5],
    [21, 0, 36, 7, 45, 2, 569]
])

linear_data = np.load(
    ROOT / "results_unsupervised_raf" / "linear_probe_results.npz"
)

unsupervised_cm = linear_data["confusion_matrix"]

# --------------------------------------------------
# 1. ACCURACY AND MACRO F1 COMPARISON
# --------------------------------------------------

names = [item["approach"] for item in comparison]
accuracies = [item["accuracy"] * 100 for item in comparison]
macro_f1 = [item["macro_f1"] * 100 for item in comparison]

x = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, accuracies, width, label="Accuracy")
bars2 = ax.bar(x + width/2, macro_f1, width, label="Macro F1")

ax.set_ylabel("Score (%)")
ax.set_title("Comparison of Learning Approaches on RAF-DB")
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax.bar_label(bars1, fmt="%.2f%%", padding=3)
ax.bar_label(bars2, fmt="%.2f%%", padding=3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "comparison_metrics.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# --------------------------------------------------
# 2. NORMALIZED CONFUSION MATRIX PLOTTER
# --------------------------------------------------

def plot_confusion_matrix(matrix, title, filename):

    row_sums = matrix.sum(axis=1, keepdims=True)

    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0
    ) * 100

    fig, ax = plt.subplots(figsize=(9, 7))

    image = ax.imshow(normalized, interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel("Predicted Emotion")
    ax.set_ylabel("Actual Emotion")

    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_yticks(np.arange(len(CLASSES)))

    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)

    plt.colorbar(image, ax=ax, label="Percentage (%)")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i,
                f"{normalized[i, j]:.1f}%",
                ha="center",
                va="center",
                fontsize=8
            )

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

# Generate all three heatmaps

plot_confusion_matrix(
    supervised_cm,
    "Supervised Learning - Normalized Confusion Matrix",
    "confusion_matrix_supervised.png"
)

plot_confusion_matrix(
    unsupervised_cm,
    "Unsupervised Learning (Linear Probe) - Normalized Confusion Matrix",
    "confusion_matrix_unsupervised.png"
)

plot_confusion_matrix(
    eacm_cm,
    "Semi-Supervised EACM - Normalized Confusion Matrix",
    "confusion_matrix_eacm.png"
)

# --------------------------------------------------
# 3. UNSUPERVISED CLUSTERING METRICS
# --------------------------------------------------

cluster_data = np.load(
    ROOT / "results_unsupervised_raf" /
    "unsupervised_clustering_results.npz",
    allow_pickle=True
)

cluster_results = cluster_data["results"]

checkpoints = [r["checkpoint"].replace(".pth", "") for r in cluster_results]

metric_names = ["ARI", "NMI", "Hungarian Accuracy"]

best_values = [
    float(cluster_results[0]["ari"]),
    float(cluster_results[0]["nmi"]),
    float(cluster_results[0]["hungarian_accuracy"])
]

latest_values = [
    float(cluster_results[1]["ari"]),
    float(cluster_results[1]["nmi"]),
    float(cluster_results[1]["hungarian_accuracy"])
]

x = np.arange(len(metric_names))

fig, ax = plt.subplots(figsize=(9, 6))

ax.bar(x - width/2, np.array(best_values) * 100, width,
       label=checkpoints[0])

ax.bar(x + width/2, np.array(latest_values) * 100, width,
       label=checkpoints[1])

ax.set_ylabel("Score (%)")
ax.set_title("Unsupervised Clustering Evaluation")
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "unsupervised_clustering_metrics.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# --------------------------------------------------
# 4. CLASS-WISE F1 RADAR CHART
# --------------------------------------------------

def calculate_class_f1(matrix):
    true_positives = np.diag(matrix)

    precision = np.divide(
        true_positives,
        matrix.sum(axis=0),
        out=np.zeros(len(CLASSES), dtype=float),
        where=matrix.sum(axis=0) != 0
    )

    recall = np.divide(
        true_positives,
        matrix.sum(axis=1),
        out=np.zeros(len(CLASSES), dtype=float),
        where=matrix.sum(axis=1) != 0
    )

    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(len(CLASSES), dtype=float),
        where=(precision + recall) != 0
    )

    return f1 * 100


# Calculate F1 for every emotion
supervised_f1 = calculate_class_f1(supervised_cm)
unsupervised_f1 = calculate_class_f1(unsupervised_cm)
eacm_f1 = calculate_class_f1(eacm_cm)

# Radar chart angles
angles = np.linspace(
    0, 2 * np.pi, len(CLASSES), endpoint=False
).tolist()

# Close each polygon
angles += angles[:1]

fig, ax = plt.subplots(
    figsize=(9, 9),
    subplot_kw={"polar": True}
)

models = [
    ("Supervised", supervised_f1),
    ("Unsupervised (Linear Probe)", unsupervised_f1),
    ("Semi-Supervised EACM", eacm_f1)
]

for name, scores in models:
    values = scores.tolist()
    values += values[:1]

    ax.plot(angles, values, linewidth=2, label=name)
    ax.fill(angles, values, alpha=0.08)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(CLASSES, fontsize=10)

ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(["20", "40", "60", "80", "100"])

ax.set_title(
    "Class-Wise F1 Score Comparison",
    pad=25,
    fontsize=14
)

ax.legend(
    loc="upper right",
    bbox_to_anchor=(1.35, 1.15)
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "classwise_f1_radar.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Generated: classwise_f1_radar.png")


# --------------------------------------------------
# FINISHED
# --------------------------------------------------

print("\n" + "=" * 60)
print("REPORT GRAPHS GENERATED SUCCESSFULLY")
print("=" * 60)

for file in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"{file.name} ({file.stat().st_size:,} bytes)")

print("\nAll graphs saved in:")
print(OUTPUT_DIR)
