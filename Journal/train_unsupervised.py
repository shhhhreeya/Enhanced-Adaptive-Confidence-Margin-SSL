import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from models.backbone import ResNet_18
from losses import SupConLoss
from dataset.raf import img_loader


class RAFUnsupervisedDataset(Dataset):
    def __init__(self, root, file_list, transform):
        self.root = root
        self.transform = transform
        self.image_paths = []

        with open(file_list, "r") as f:
            for line in f:
                image_path = line.strip().split()[0]
                self.image_paths.append(image_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = os.path.join(self.root, self.image_paths[index])
        image = img_loader(path)

        view1 = self.transform(image)
        view2 = self.transform(image)

        return view1, view2


class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, projection_dim=128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        return nn.functional.normalize(self.layers(x), dim=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.00025)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--out", type=str, default="../results_unsupervised_raf")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. This experiment expects a CUDA GPU.")

    device = torch.device("cuda")
    print("Device:", torch.cuda.get_device_name(0))

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            )
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = RAFUnsupervisedDataset(
        root="../datasets/DATASET/train",
        file_list="../RAF_train_label2.txt",
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    print("Training images:", len(dataset))
    print("Batches per epoch:", len(loader))

    backbone = ResNet_18(num_classes=7).to(device)
    projection = ProjectionHead().to(device)

    optimizer = optim.Adam(
        list(backbone.parameters()) + list(projection.parameters()),
        lr=args.lr
    )

    criterion = SupConLoss(temperature=0.07)

    os.makedirs(args.out, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(args.epochs):
        backbone.train()
        projection.train()

        running_loss = 0.0

        for batch_idx, (view1, view2) in enumerate(loader):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            images = torch.cat([view1, view2], dim=0)

            optimizer.zero_grad(set_to_none=True)

            _, features = backbone(images)
            z1, z2 = projection(features).chunk(2, dim=0)

            contrastive_features = torch.stack([z1, z2], dim=1)
            loss = criterion(contrastive_features)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch + 1}/{args.epochs}] "
                    f"Batch [{batch_idx + 1}/{len(loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        epoch_loss = running_loss / len(loader)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Average Loss: {epoch_loss:.4f}"
        )

        checkpoint = {
            "epoch": epoch + 1,
            "backbone_state_dict": backbone.state_dict(),
            "projection_state_dict": projection.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "seed": args.seed
        }

        torch.save(
            checkpoint,
            os.path.join(args.out, "unsupervised_latest.pth")
        )

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                checkpoint,
                os.path.join(args.out, "unsupervised_best.pth")
            )

    print("Training complete.")
    print("Best loss:", best_loss)


if __name__ == "__main__":
    main()
