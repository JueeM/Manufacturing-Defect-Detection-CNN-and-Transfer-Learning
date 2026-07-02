"""
03_transfer_learning.py
Fine-tunes a pretrained ResNet-18 on the defect dataset.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from utils import get_device, load_datasets, get_dataloaders, get_class_weights, save_checkpoint

DATA_DIR = r"C:\Users\Admin\Desktop\PROJECTS\defect_detection\archive\casting_data\casting_data"
print("DATA_DIR is:", DATA_DIR)
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4


def build_model(num_classes, device, unfreeze_layer4=True):
    model = models.resnet18(weights="IMAGENET1K_V1")

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Replace final classification layer (always trainable)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Optionally unfreeze the last residual block for better fine-tuning
    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    return model.to(device)


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

    model = build_model(num_classes=len(train_ds.classes), device=device, unfreeze_layer4=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Only optimize parameters that require grad (fc + layer4)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

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
            save_checkpoint(model, "../results/resnet18_best.pt")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal transfer learning (ResNet-18) test accuracy: {test_acc:.4f}")

    with open("../results/transfer_history.json", "w") as f:
        json.dump({**history, "test_acc": test_acc}, f, indent=2)


if __name__ == "__main__":
    main()