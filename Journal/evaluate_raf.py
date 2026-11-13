import argparse
import os
import torch
import numpy as np
import torchvision.transforms as transforms

from models.backbone import ResNet_18
from dataset.raf import Dataset_RAF
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--test-root", required=True)
parser.add_argument("--label-test", required=True)
parser.add_argument("--num-workers", type=int, default=0)
args = parser.parse_args()

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available.")

device = torch.device("cuda")

class_names = [
    "Surprise", "Fear", "Disgust", "Happiness",
    "Sadness", "Anger", "Neutral"
]

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform_val = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

test_set = Dataset_RAF(
    args.test_root,
    args.label_test,
    transform=transform_val
)

test_loader = DataLoader(
    test_set,
    batch_size=128,
    shuffle=False,
    num_workers=args.num_workers
)

model = torch.nn.DataParallel(ResNet_18(num_classes=7)).to(device)

checkpoint = torch.load(
    args.checkpoint,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["ema_state_dict"])
model.eval()

print("Loaded checkpoint:", args.checkpoint)
print("Checkpoint epoch:", checkpoint["epoch"])
print("Checkpoint best accuracy:", checkpoint["best_acc"])
print("Test images:", len(test_set))

confusion = torch.zeros((7, 7), dtype=torch.int64)

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs, _ = model(inputs)
        predictions = outputs.argmax(dim=1)

        encoded = targets * 7 + predictions
        confusion += torch.bincount(
            encoded.cpu(), minlength=49
        ).reshape(7, 7)

confusion = confusion.numpy()

print("\n========== CONFUSION MATRIX ==========")
print("Rows = Actual | Columns = Predicted")
print("Classes:", class_names)
print(confusion)

total_correct = np.trace(confusion)
total = confusion.sum()

print("\n========== CLASS-WISE METRICS ==========")
print(f"Overall accuracy: {100 * total_correct / total:.2f}%")

for i, name in enumerate(class_names):
    tp = confusion[i, i]
    actual = confusion[i, :].sum()
    predicted = confusion[:, i].sum()

    precision = tp / predicted if predicted else 0.0
    recall = tp / actual if actual else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    print(
        f"{name:12s} | "
        f"Correct: {tp:4d}/{actual:4d} | "
        f"Precision: {precision * 100:6.2f}% | "
        f"Recall: {recall * 100:6.2f}% | "
        f"F1: {f1 * 100:6.2f}%"
    )

print("======================================")
