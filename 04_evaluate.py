"""
04_evaluate.py
Loads both trained models, generates confusion matrices, classification
reports (precision/recall/F1 per class), and a side-by-side comparison
plot. Run this after both 02_baseline_cnn.py and 03_transfer_learning.py.
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from torchvision import models
import torch.nn as nn

from utils import get_device, load_datasets, get_dataloaders
from importlib import import_module

DATA_DIR = r"C:\Users\Admin\Desktop\PROJECTS\defect_detection\archive\casting_data\casting_data"
IMG_SIZE = 128
BATCH_SIZE = 32

# Import SimpleCNN class from the baseline script
baseline_module = import_module("02_baseline_cnn") if False else None
# (direct import above is awkward due to filename starting with a digit;
#  redefine the class here instead so this script is standalone)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2, img_size=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        )
        reduced = img_size // 16
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * reduced * reduced, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion(y_true, y_pred, class_names, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")
    return cm


def main():
    device = get_device()
    _, test_ds = load_datasets(DATA_DIR, img_size=IMG_SIZE, augment=False)
    _, _, test_loader = get_dataloaders(test_ds, test_ds, batch_size=BATCH_SIZE, val_split=0.0)
    # ^ using test_ds twice is fine here since val_split=0 means no val split is taken

    class_names = test_ds.classes
    num_classes = len(class_names)

    os.makedirs("../results", exist_ok=True)
    summary = {}

    # ---- Baseline CNN ----
    baseline_path = "../results/baseline_cnn_best.pt"
    if os.path.exists(baseline_path):
        baseline_model = SimpleCNN(num_classes=num_classes, img_size=IMG_SIZE).to(device)
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=device))
        y_true, y_pred = get_predictions(baseline_model, test_loader, device)

        print("\n=== Baseline CNN ===")
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        print(classification_report(y_true, y_pred, target_names=class_names))
        plot_confusion(y_true, y_pred, class_names, "Baseline CNN — Confusion Matrix",
                        "../results/confusion_baseline.png")
        summary["baseline_cnn"] = {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {c: report[c] for c in class_names}
        }
    else:
        print(f"Skipping baseline — {baseline_path} not found. Run 02_baseline_cnn.py first.")

    # ---- Transfer learning (ResNet-18) ----
    resnet_path = "../results/resnet18_best.pt"
    if os.path.exists(resnet_path):
        resnet_model = models.resnet18(weights=None)
        resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)
        resnet_model.load_state_dict(torch.load(resnet_path, map_location=device))
        resnet_model = resnet_model.to(device)
        y_true, y_pred = get_predictions(resnet_model, test_loader, device)

        print("\n=== Transfer Learning (ResNet-18) ===")
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        print(classification_report(y_true, y_pred, target_names=class_names))
        plot_confusion(y_true, y_pred, class_names, "ResNet-18 (Fine-tuned) — Confusion Matrix",
                        "../results/confusion_resnet.png")
        summary["resnet18_transfer"] = {
            "accuracy": report["accuracy"],
            "macro_f1": report["macro avg"]["f1-score"],
            "per_class": {c: report[c] for c in class_names}
        }
    else:
        print(f"Skipping ResNet-18 — {resnet_path} not found. Run 03_transfer_learning.py first.")

    # ---- Comparison bar chart ----
    if "baseline_cnn" in summary and "resnet18_transfer" in summary:
        models_list = ["Baseline CNN", "ResNet-18 (Transfer)"]
        accs = [summary["baseline_cnn"]["accuracy"], summary["resnet18_transfer"]["accuracy"]]
        f1s = [summary["baseline_cnn"]["macro_f1"], summary["resnet18_transfer"]["macro_f1"]]

        x = np.arange(len(models_list))
        width = 0.35
        plt.figure(figsize=(6, 4))
        plt.bar(x - width/2, accs, width, label="Accuracy")
        plt.bar(x + width/2, f1s, width, label="Macro F1")
        plt.xticks(x, models_list)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.title("Baseline CNN vs Transfer Learning")
        plt.tight_layout()
        plt.savefig("../results/model_comparison.png")
        print("Saved ../results/model_comparison.png")

        # Identify hardest class (lowest recall) for the best model
        best_model_name = "resnet18_transfer" if accs[1] >= accs[0] else "baseline_cnn"
        per_class = summary[best_model_name]["per_class"]
        hardest_class = min(class_names, key=lambda c: per_class[c]["recall"])
        print(f"\nHardest class to classify (lowest recall, best model): "
              f"{hardest_class} (recall={per_class[hardest_class]['recall']:.3f})")
        summary["hardest_class"] = hardest_class

    with open("../results/evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved ../results/evaluation_summary.json")


if __name__ == "__main__":
    main()