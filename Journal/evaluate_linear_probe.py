import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from models.backbone import ResNet_18
from dataset.raf import Dataset_RAF, Dataset_RAF_labeled


parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint",
    default="../results_unsupervised_raf/unsupervised_best.pth"
)
parser.add_argument(
    "--split",
    default="../results_raf_20_epochs/raf_split_seed2.npz"
)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--seed", type=int, default=2)
args = parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_features(model, loader, device):
    features_list = []
    labels_list = []

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            _, features = model(images)

            features_list.append(features.cpu())
            labels_list.append(labels.cpu())

    return (
        torch.cat(features_list),
        torch.cat(labels_list).long()
    )


def main():
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

    device = torch.device("cuda")
    print("Device:", device)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    split = np.load(args.split)
    labeled_indices = split["labeled_indices"]

    train_dataset = Dataset_RAF_labeled(
        "../datasets/DATASET/train",
        "../RAF_train_label2.txt",
        labeled_indices,
        transform=transform
    )

    test_dataset = Dataset_RAF(
        "../datasets/DATASET/test",
        "../RAF_test_label2.txt",
        transform=transform
    )

    print("Labeled training images:", len(train_dataset))
    print("Test images:", len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    backbone = ResNet_18(num_classes=7).to(device)
    backbone.load_state_dict(checkpoint["backbone_state_dict"])

    # Freeze the entire unsupervised backbone.
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    print("\nExtracting training features...")
    train_features, train_labels = extract_features(
        backbone, train_loader, device
    )

    print("Extracting test features...")
    test_features, test_labels = extract_features(
        backbone, test_loader, device
    )

    print("Training feature shape:", tuple(train_features.shape))
    print("Test feature shape:", tuple(test_features.shape))

    # Train only a new linear classification layer.
    feature_dim = train_features.shape[1]

    classifier = nn.Linear(feature_dim, 7).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=args.lr
    )

    feature_dataset = TensorDataset(
        train_features,
        train_labels
    )

    feature_loader = DataLoader(
        feature_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    print("\n===== UNSUPERVISED LINEAR PROBE =====")
    print("Backbone: Frozen")
    print("Classifier: Linear layer")
    print("Training epochs:", args.epochs)

    for epoch in range(args.epochs):
        classifier.train()
        running_loss = 0.0

        for features, labels in feature_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = classifier(features)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = running_loss / len(feature_loader)
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"Loss: {avg_loss:.4f}"
            )

    # Evaluate only after training is complete.
    classifier.eval()

    with torch.no_grad():
        test_logits = classifier(test_features.to(device))
        predictions = test_logits.argmax(dim=1).cpu().numpy()

    y_true = test_labels.numpy()

    accuracy = accuracy_score(y_true, predictions)
    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=np.arange(7)
    )

    class_names = [
        "Surprise", "Fear", "Disgust", "Happiness",
        "Sadness", "Anger", "Neutral"
    ]

    print("\n===== LINEAR PROBE RESULTS =====")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            predictions,
            labels=np.arange(7),
            target_names=class_names,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(matrix)

    output_dir = "../results_unsupervised_raf"
    os.makedirs(output_dir, exist_ok=True)

    torch.save(
        {
            "classifier_state_dict": classifier.state_dict(),
            "feature_dim": feature_dim,
            "epoch": args.epochs,
            "seed": args.seed,
            "backbone_checkpoint": args.checkpoint,
            "accuracy": accuracy,
            "macro_f1": macro_f1
        },
        os.path.join(output_dir, "linear_probe_classifier.pth")
    )

    np.savez_compressed(
        os.path.join(output_dir, "linear_probe_results.npz"),
        accuracy=accuracy,
        macro_f1=macro_f1,
        predictions=predictions,
        labels=y_true,
        confusion_matrix=matrix
    )

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
