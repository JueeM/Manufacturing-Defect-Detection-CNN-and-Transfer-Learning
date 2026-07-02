"""
02_baseline_cnn.py
Trains a small CNN from scratch as the baseline model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_device, load_datasets, get_dataloaders, get_class_weights, save_checkpoint

DATA_DIR = r"C:\Users\Admin\Desktop\PROJECTS\defect_detection\archive\casting_data\casting_data"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-3


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, img_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        reduced = img_size // 16  # 4 maxpools of stride 2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * reduced * reduced, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


def main():
    device = get_device()
    print("Using device:", device)

    train_ds, test_ds = load_datasets(DATA_DIR, img_size=IMG_SIZE, augment=True)
    train_loader, val_loader, test_loader = get_dataloaders(train_ds, test_ds, batch_size=BATCH_SIZE)

    class_weights = get_class_weights(train_ds).to(device)
    print("Class weights (imbalance correction):", class_weights, train_ds.classes)

    model = SimpleCNN(num_classes=len(train_ds.classes), img_size=IMG_SIZE).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("../results", exist_ok=True)
            save_checkpoint(model, "../results/baseline_cnn_best.pt")

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal baseline CNN test accuracy: {test_acc:.4f}")

    import json
    with open("../results/baseline_history.json", "w") as f:
        json.dump({**history, "test_acc": test_acc}, f, indent=2)


if __name__ == "__main__":
    main()