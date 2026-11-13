import os
import sys
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.backbone import ResNet_18
from dataset.raf import Dataset_RAF_labeled, Dataset_RAF

parser = argparse.ArgumentParser()
parser.add_argument("--train-root", default="./datasets/DATASET/train")
parser.add_argument("--label-train", default="./RAF_train_label2.txt")
parser.add_argument("--test-root", default="./datasets/DATASET/test")
parser.add_argument("--label-test", default="./RAF_test_label2.txt")
parser.add_argument("--split", default="./results_raf_20_epochs/raf_split_seed2.npz")
parser.add_argument("--out", default="./results_supervised_raf")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--iterations-per-epoch", type=int, default=900)
parser.add_argument("--lr", type=float, default=0.00025)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--seed", type=int, default=2)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this experiment.")

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

device = torch.device("cuda")
os.makedirs(args.out, exist_ok=True)

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomApply([
        transforms.RandomCrop(224, padding=8)
    ], p=0.5),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

split = np.load(args.split)
labeled_indices = split["labeled_indices"]

train_set = Dataset_RAF_labeled(
    args.train_root,
    args.label_train,
    labeled_indices,
    transform=transform_train
)

test_set = Dataset_RAF(
    args.test_root,
    args.label_test,
    transform=transform_val
)

if len(train_set) != 4000:
    raise RuntimeError(f"Expected 4000 labeled images, got {len(train_set)}")

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True
)

test_loader = DataLoader(
    test_set,
    batch_size=128,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True
)

model = nn.DataParallel(ResNet_18(num_classes=7)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

best_acc = 0.0
steps_per_epoch = args.iterations_per_epoch

print("\n===== SUPERVISED RAF-DB EXPERIMENT =====")
print("Training images:", len(train_set))
print("Testing images:", len(test_set))
print("Epochs:", args.epochs)
print("Iterations per epoch:", steps_per_epoch)
print("Device:", device)

for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    train_iter = iter(train_loader)

    for step in range(steps_per_epoch):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs, _ = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % 100 == 0:
            print(
                f"Epoch {epoch}/{args.epochs} "
                f"Step {step + 1}/{steps_per_epoch} "
                f"Loss {running_loss / (step + 1):.4f}"
            )

    train_accuracy = 100.0 * correct / total

    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs, _ = model(images)
            predictions = outputs.argmax(dim=1)

            test_correct += (predictions == labels).sum().item()
            test_total += labels.size(0)

    test_accuracy = 100.0 * test_correct / test_total
    average_loss = running_loss / steps_per_epoch

    print(
        f"\nEpoch {epoch}: "
        f"Loss={average_loss:.4f} "
        f"Train Acc={train_accuracy:.2f}% "
        f"Test Acc={test_accuracy:.2f}%"
    )

    is_best = test_accuracy > best_acc
    best_acc = max(best_acc, test_accuracy)

    checkpoint = {
        "epoch": epoch,
        "best_acc": best_acc,
        "ema_state_dict": model.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "seed": args.seed,
        "labeled_indices": labeled_indices,
    }

    torch.save(
        checkpoint,
        os.path.join(args.out, "supervised_checkpoint.pth.tar")
    )

    if is_best:
        torch.save(
            checkpoint,
            os.path.join(args.out, "supervised_model_best.pth.tar")
        )
        print("New best checkpoint saved!")

print("\n===== TRAINING COMPLETE =====")
print(f"Best test accuracy: {best_acc:.2f}%")
print("Results directory:", args.out)
