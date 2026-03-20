# ML4SCI DeepLense - GSoC 2026 - Test I: Multi-Class Classification

## Overview

This experiment addresses **multi-class classification of gravitational lensing simulations** into three categories:

- **no** - No substructure
- **sphere** - Subhalo (spherical perturbation)
- **vort** - Vortex perturbation

The model uses a **transfer learning approach with a ResNet50 backbone pretrained on ImageNet**.

---

## Model Architecture

- Backbone: **ResNet50**
- Pretraining: ImageNet
- Classifier head:
  - Dropout (0.5)
  - Fully connected layer -> 3 classes
- Total parameters: **23,514,179**

---

## Training Configuration

| Parameter     | Value                      |
| ------------- | -------------------------- |
| Batch size    | 128                        |
| Epochs        | 10                         |
| Learning rate | 3e-4                       |
| Weight decay  | 1e-4                       |
| Optimizer     | AdamW                      |
| Scheduler     | CosineAnnealingLR (eta_min=1e-6) |
| Grad clipping | clip_grad_norm\_(1.0)      |
| Hardware      | NVIDIA RTX 5060 Laptop GPU |

---

## Dataset

Dataset provided by the organizers (pre-split).

| Split      | Samples |
| ---------- | ------- |
| Training   | 30,000  |
| Validation | 7,500   |

Total samples: **37,500**

---

## Final Performance

### Validation Accuracy

**93.88%** (best epoch: 10)

### ROC-AUC Scores

| Class         | AUC        |
| ------------- | ---------- |
| no            | 0.9908     |
| sphere        | 0.9816     |
| vort          | 0.9926     |
| Macro Average | **0.9883** |
| Micro Average | **0.9902** |

### Classification Metrics

| Class  | Precision | Recall | F1     |
| ------ | --------- | ------ | ------ |
| no     | 0.9276    | 0.9688 | 0.9478 |
| sphere | 0.9344    | 0.9004 | 0.9171 |
| vort   | 0.9548    | 0.9472 | 0.9510 |

Overall macro F1: **0.9386**

### Confusion Matrix (Normalised)

| True \ Pred | no     | sphere | vort   |
| ----------- | ------ | ------ | ------ |
| no          | 96.88% | 2.68%  | 0.44%  |
| sphere      | 5.92%  | 90.04% | 4.04%  |
| vort        | 1.64%  | 3.64%  | 94.72% |

---

## Included Files

| File                        | Description                    |
| --------------------------- | ------------------------------ |
| `results_summary.json`      | Complete experiment metrics    |
| `roc_curves.png`            | Multi-class ROC curves         |
| `roc_curves_individual.png` | ROC curves per class           |
| `confusion_matrix.png`      | Confusion matrix visualization |
| `training_history.png`      | Training/validation curves     |

---

## Model Weights

Due to GitHub file size limits, trained model weights are hosted on Google Drive.

Download here: https://drive.google.com/drive/folders/1i4v1g0ihsrAGYysGp9y__CpRq7OK-MWo?usp=sharing

File: `model_weights.pth` is located inside the `test_1` folder.

## Loading the Model

```python
import torch
import torchvision.models as models
import torch.nn as nn

# Build model
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_features, 3)
)

# Load checkpoint
checkpoint = torch.load("model_weights.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

---

## Notes

- The model checkpoint corresponds to the **best validation epoch** (epoch 10).
- The dataset is provided pre-split at 80:20 (train/val). The provided split is used as-is.
- This notebook serves as the **baseline** for Test VII (PINN), which uses identical training configuration for a fair comparison.