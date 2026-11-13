import os
import numpy as np
import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from scipy.optimize import linear_sum_assignment

from models.backbone import ResNet_18
from dataset.raf import Dataset_RAF


def extract_features(model, loader, device):
    all_features = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            _, features = model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def clustering_accuracy(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(7))
    row_ind, col_ind = linear_sum_assignment(matrix.max() - matrix)
    return matrix[row_ind, col_ind].sum() / matrix.sum()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = Dataset_RAF(
        "../datasets/DATASET/train",
        "../RAF_train_label2.txt",
        transform=transform
    )
    test_dataset = Dataset_RAF(
        "../datasets/DATASET/test",
        "../RAF_test_label2.txt",
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    results_dir = "../results_unsupervised_raf"
    checkpoint_names = ["unsupervised_best.pth", "unsupervised_latest.pth"]

    results = []

    for checkpoint_name in checkpoint_names:
        checkpoint_path = os.path.join(results_dir, checkpoint_name)

        if not os.path.exists(checkpoint_path):
            print("Checkpoint not found:", checkpoint_path)
            continue

        print("\nEvaluating:", checkpoint_name)

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = ResNet_18(num_classes=7).to(device)
        model.load_state_dict(checkpoint["backbone_state_dict"])
        model.eval()

        train_features, train_labels = extract_features(
            model, train_loader, device
        )
        test_features, test_labels = extract_features(
            model, test_loader, device
        )

        print("Train embeddings:", train_features.shape)
        print("Test embeddings:", test_features.shape)

        # Fit clusters on training embeddings only.
        # Test labels are used only for evaluation, never for fitting.
        kmeans = KMeans(
            n_clusters=7,
            random_state=2,
            n_init=20
        )
        kmeans.fit(train_features)
        test_clusters = kmeans.predict(test_features)

        ari = adjusted_rand_score(test_labels, test_clusters)
        nmi = normalized_mutual_info_score(test_labels, test_clusters)
        matched_acc = clustering_accuracy(test_labels, test_clusters)

        cluster_counts = np.bincount(test_clusters, minlength=7)

        print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
        print(f"ARI: {ari:.4f}")
        print(f"NMI: {nmi:.4f}")
        print(f"Hungarian-matched accuracy: {matched_acc * 100:.2f}%")
        print("Test cluster distribution:", cluster_counts.tolist())

        results.append({
            "checkpoint": checkpoint_name,
            "epoch": checkpoint.get("epoch", -1),
            "ari": ari,
            "nmi": nmi,
            "hungarian_accuracy": matched_acc,
            "cluster_counts": cluster_counts
        })

    if results:
        output_path = os.path.join(
            results_dir, "unsupervised_clustering_results.npz"
        )
        np.savez(output_path, results=np.array(results, dtype=object))
        print("\nSaved results to:", output_path)
    else:
        print("\nNo checkpoints were evaluated.")


if __name__ == "__main__":
    main()
