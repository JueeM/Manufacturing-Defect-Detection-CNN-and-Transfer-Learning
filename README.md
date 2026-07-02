# Manufacturing-Defect-Detection-CNN-and-Transfer-Learning

# Manufacturing Defect Detection — CNN & Transfer Learning

Automated visual quality inspection pipeline for casting manufacturing parts, built using PyTorch. Compares a CNN trained from scratch against a fine-tuned ResNet-18 (transfer learning) on 7,000+ real industrial images.

> **Motivation:** Manual quality inspection on production lines is slow, inconsistent, and expensive. This project explores whether a deep learning model can reliably flag defective casting parts from a single image — a directly deployable solution for manufacturing automation.

---

## Results

| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Baseline CNN (from scratch) | 100% | 1.00 |
| ResNet-18 (Transfer Learning) | 99.72% | 1.00 |

Both models achieved near-perfect classification on 715 held-out test images, with perfect per-class precision and recall — confirming strong visual separability between defective and ok casting parts.

---

## Dataset

**Casting Product Image Data for Quality Inspection** — [Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)

- 7,348 total images of industrial casting parts (pump impellers)
- 2 classes: `def_front` (defective) and `ok_front` (ok)
- Pre-split into train (6,633 images) and test (715 images)
- Class imbalance: 3,758 defective vs 2,875 ok in training set

---

## Project Structure

```
defect-detection/
├── archive/casting_data/casting_data/   ← dataset (train/ and test/ folders)
├── src/
│   ├── utils.py                          ← shared: data loading, transforms, class weights
│   ├── 01_explore.ipynb                  ← EDA: class distribution, sample image grid
│   ├── 02_baseline_cnn.py                ← CNN trained from scratch
│   ├── 03_transfer_learning.py           ← fine-tuned ResNet-18
│   └── 04_evaluate.py                    ← confusion matrix, classification report, comparison chart
├── results/
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── confusion_baseline.png
│   ├── confusion_resnet.png
│   ├── model_comparison.png
│   └── evaluation_summary.json
└── README.md
```

---

## Approach

### 1. Data Exploration
Analyzed class distribution, confirmed imbalance (3,758 defective vs 2,875 ok), and visualized sample images from both classes to verify labels and visual distinction.

### 2. Handling Class Imbalance
Used two complementary strategies:
- **Inverse-frequency class weighting** in `CrossEntropyLoss` — penalizes mistakes on the minority class more heavily during training
- **Data augmentation** on training set only (random flips, rotation ±15°, color jitter) — exposes the model to varied versions of each image across epochs, reducing overfitting without increasing dataset size

### 3. Baseline CNN (from scratch)
Small 4-block CNN architecture:
- 4 × (Conv2d → BatchNorm → ReLU → MaxPool) blocks, doubling channels each time (16 → 32 → 64 → 128)
- Fully connected classifier with Dropout(0.4) for regularization
- Trained for 20 epochs with Adam optimizer and ReduceLROnPlateau scheduler

### 4. Transfer Learning (ResNet-18)
Fine-tuned a pretrained ResNet-18 (ImageNet weights):
- Froze all early layers — kept generic feature extractors (edges, textures) intact
- Unfroze `layer4` + final `fc` layer for domain-specific fine-tuning
- Lower learning rate (1e-4) to avoid overwriting pretrained weights
- Trained for 15 epochs — faster convergence than baseline due to pretrained features

### 5. Evaluation
- Confusion matrix per model
- Per-class precision, recall, F1-score
- Side-by-side accuracy and macro F1 comparison chart
- Identified hardest-to-classify class by lowest recall

---

## Key Findings

- The baseline CNN achieved 100% test accuracy, suggesting the visual distinction between defective and ok casting parts is learnable even with a simple architecture
- Transfer learning (ResNet-18) confirmed this with 99.72% accuracy and faster convergence, reaching high val accuracy in fewer epochs due to pretrained ImageNet features
- Both models achieved perfect per-class precision and recall — no defective parts were misclassified as ok, which is the highest-stakes error in a real manufacturing context

---

## Setup & Usage

### Install dependencies
```bash
pip install torch torchvision scikit-learn matplotlib seaborn pandas
```

### Run in order
```bash
cd src
python 01_explore.py             # EDA and visualizations
python 02_baseline_cnn.py        # train baseline CNN
python 03_transfer_learning.py   # fine-tune ResNet-18
python 04_evaluate.py            # generate all evaluation outputs
```

### Update DATA_DIR
In each script, set `DATA_DIR` to point at the folder containing your `train/` and `test/` subfolders:
```python
DATA_DIR = r"C:\your\path\to\casting_data"  # Windows
DATA_DIR = "../data/casting_data"            # Mac/Linux
```

---

## Limitations & Next Steps

- **Dataset is relatively easy** — strong visual contrast between classes means even a simple CNN saturates near 100%. A more realistic challenge would be the [NEU Surface Defect Database](https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database) (6 subtle defect classes on steel surfaces)
- **Binary classification only** — real production lines often require multi-class defect categorization (crack vs. scratch vs. inclusion)
- **Edge deployment** — swap ResNet-18 for MobileNetV2 to explore the accuracy vs. inference speed tradeoff for real-time line-side inspection
- **Focal loss** — alternative to class weighting for imbalanced data, worth benchmarking on a harder dataset

---

## Technologies
Python · PyTorch · torchvision · scikit-learn · OpenCV · Matplotlib · Seaborn

---

*Project by Juee Mahajan — Mechanical Engineering, NITK Surathkal*
