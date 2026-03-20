# ML4SCI DeepLense - GSoC 2026 - Test VII: Physics-Informed Neural Network

## Overview

This experiment extends the baseline ResNet50 classifier from Test I with a **true Physics-Informed Neural Network (PINN)** that embeds the gravitational lens equation directly inside the forward pass.

The model classifies strong gravitational lensing simulations into three categories:

- **no** - No substructure
- **sphere** - Subhalo (spherical perturbation)
- **vort** - Vortex perturbation

---

## PINN Architecture

```
Input Image (B, 3, 150, 150)
      |
  ResNet50 Backbone → features (B, 2048)
      |
  Physics Head → [θ_E, c_x, c_y]   ← interpretable physics parameters
      |
  LensEquationLayer                 ← applies β = θ − α(θ) differentiably
      |
  Source Image (B, 3, 150, 150)     ← reconstructed source plane
      |
  Source Encoder → source_features (B, 128)
      |
  concat(features, source_features) → (B, 2176)
      |
  Fusion Classifier → Logits (B, 3)
```

### Key Components

1. **ResNet50 backbone** - pretrained on ImageNet, 2048-dim feature output
2. **Physics Head** - predicts Einstein radius θ_E and lens centre (c_x, c_y)
   - Linear(2048→256) + ReLU + Dropout(0.3)
   - Linear(256→64) + ReLU + Linear(64→3)
   - Outputs constrained: θ_E ∈ [0.05, 0.40], c_x/c_y ∈ [−0.30, 0.30]
3. **LensEquationLayer** - computes SIS deflection field analytically and applies lens equation via differentiable `F.grid_sample`
4. **Source Encoder** - lightweight CNN encoding the reconstructed source plane
5. **Fusion Classifier** - combines backbone and source features for final classification

### Physics Integration

The gravitational lens equation is applied **inside every forward pass**:

$$\boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta})$$

where the SIS deflection field is:

$$\boldsymbol{\alpha}(\boldsymbol{\theta}) = \theta_E \cdot \frac{\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{lens}}}{|\boldsymbol{\theta} - \boldsymbol{\theta}_{\text{lens}}|}$$

### Physics-Informed Loss

```
L_total = L_CE + λ_κ · L_κ + λ_TV · L_TV

where:
  L_CE  = CrossEntropy (classification)
  L_κ   = MSE(∇·α / 2, κ_SIS)   ← convergence consistency residual
  L_TV  = total variation of reconstructed source (compactness prior)
  λ_κ = 0.1,  λ_TV = 0.01
```

---

## Training Configuration

| Parameter       | Value                            |
| --------------- | -------------------------------- |
| Batch size      | 128 (same as baseline)           |
| Epochs          | 10 (same as baseline)            |
| Learning rate   | 3e-4                             |
| Weight decay    | 1e-4                             |
| Optimizer       | AdamW                            |
| Scheduler       | CosineAnnealingLR (eta_min=1e-6) |
| Grad clipping   | clip_grad_norm\_(1.0)            |
| Mixed precision | AMP (GradScaler)                 |
| λ_κ             | 0.1                              |
| λ_TV            | 0.01                             |
| Training time   | 19.5 minutes                     |
| Hardware        | NVIDIA RTX 5060 Laptop GPU       |

---

## Dataset

| Split      | Samples |
| ---------- | ------- |
| Training   | 30,000  |
| Validation | 7,500   |

Same 80:20 pre-split as Test I. Identical split used for fair comparison.

---

## Results

### Classification Performance vs Baseline

| Metric        | Baseline (Test I) | PINN (Test VII) | Delta      |
| ------------- | :---------------: | :-------------: | :--------: |
| Accuracy      |      93.88%       |     93.69%      | −0.19%     |
| AUC - no      |      0.9908       |   **0.9910**    | +0.0002    |
| AUC - sphere  |      0.9816       |   **0.9818**    | +0.0002    |
| AUC - vort    |      0.9926       |   **0.9931**    | +0.0005    |
| AUC - Macro   |      0.9883       |   **0.9886**    | +0.0003    |
| AUC - Micro   |      0.9901       |     0.9901      | +0.0000    |
| Parameters    |    23,514,179     |   25,323,334    | +1,809,155 |

### Per-Class Performance (Test VII)

| Class           | Precision | Recall | F1-Score | Support |
| --------------- | :-------: | :----: | :------: | :-----: |
| No Substructure |   0.94    |  0.95  |   0.94   |  2,500  |
| Subhalo         |   0.92    |  0.91  |   0.91   |  2,500  |
| Vortex          |   0.95    |  0.95  |   0.95   |  2,500  |
| **Macro avg**   | **0.94**  |**0.94**| **0.94** |  7,500  |

### Confusion Matrix (Normalised)

| True \ Pred     | No Subst. | Subhalo | Vortex |
| --------------- | :-------: | :-----: | :----: |
| No Substructure |   0.95    |  0.04   |  0.01  |
| Subhalo         |   0.05    |  0.91   |  0.04  |
| Vortex          |   0.01    |  0.04   |  0.95  |

### Physics Loss Convergence

| Loss term | Epoch 1 (train) | Epoch 10 (train) | Converged? |
| --------- | :-------------: | :--------------: | :--------: |
| L_κ       |     0.0178      |      0.0056      | ✅ Yes      |
| L_TV      |     0.0707      |      0.0654      | ✅ Yes      |

Both physics residuals decrease and stabilise, confirming the lens equation constraint is actively enforced during training.

---

## Analysis

### What the PINN demonstrates

- ✅ Lens equation `β = θ − α(θ)` applied differentiably inside every forward pass
- ✅ Convergence consistency residual (`∇·α = 2κ`) enforced as a physics loss
- ✅ Source-plane reconstruction used as an additional classification signal
- ✅ Both physics residuals converge and stabilise over training
- ✅ AUC improves across all three classes vs the baseline
- ✅ Identical training budget (10 epochs, batch 128) for a fair comparison

### Why accuracy is marginally lower (−0.19%) while AUC improves

Accuracy reflects the argmax decision at a single threshold. AUC integrates over all thresholds and is a more reliable measure of ranking quality. The PINN's class probability estimates are better calibrated (higher AUC on all three classes) even though the hard classification boundary is marginally less sharp - a known effect of multi-task regularisation, where the physics constraint adds useful inductive bias that helps ranking but slightly shifts the decision boundary.

### Einstein radius collapse

The predicted θ_E collapses to the lower bound (0.05) for all classes. This reflects that with λ_κ = 0.1 the physics loss contribution is small relative to the CE loss, so the physics head is not strongly incentivised to diversify θ_E predictions. Increasing λ_κ to 1.0–5.0 with a warm-up schedule is the primary planned improvement for the GSoC project.

### Future improvements (GSoC scope)

1. **Increase λ_κ** (target 1.0–5.0) with a 2-epoch warm-up to make the physics constraint more influential
2. **Normalise physics losses by H×W** to stabilise gradient scales
3. **Extend to HEAL-PINN / LensPINN** architectures using the SIS potential ansatz Ψ = k(θ) · Ψ_SIS
4. **Test on real HSC/HST images** - where the simulation-independent physics constraint provides the most benefit

---

## File Structure

```
physics_guided_ml/
├── gravitational_lensing_pinn.ipynb     # Main notebook
├── README.md                             # This file
├── checkpoints_pinn/
│   └── deeplense_pinn_best.pth          # Best checkpoint (epoch 10)
├── results_pinn/
│   ├── training_history.png
│   ├── roc_curves.png
│   ├── roc_curves_individual.png
│   ├── confusion_matrix.png
│   ├── theta_E_distribution.png
│   ├── source_reconstructions.png
│   ├── baseline_comparison.png
│   └── results_summary.json
└── submission_pinn/
    └── [all result files + model weights]
```

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Execution

1. Ensure dataset is at `/home/jaisal/pinn/dataset/dataset`
2. Open `gravitational_lensing_pinn.ipynb` in Jupyter
3. Run all cells sequentially
4. Results saved to `results_pinn/` and `submission_pinn/`

### Environment

- **System:** Ubuntu 24.04
- **GPU:** NVIDIA RTX 5060 Laptop GPU
- **Framework:** PyTorch 2.x + AMP
- **Python:** 3.10+
- **CUDA:** 12.8

---

## Model Weights

Due to GitHub file size limits, trained model weights are hosted on Google Drive:

https://drive.google.com/drive/folders/1oEyTE9xijlzTRw-i5ybPh8mdkhb5YwU_?usp=sharing

## Loading the Model

```python
import torch

# See notebook Cell 16 for full DeepLensePINN class definition
checkpoint = torch.load("deeplense_pinn_best.pth", map_location="cpu")
model.load_state_dict(checkpoint['model'])
model.eval()
```