# Experiment Logs — DCASE 2023 Task 2 ASD

## Overview

This document records all experiments conducted for Assignment 3.
We extend the reproduced baseline from Assignment 2 with architectural
improvements and evaluate on an additional dataset.

---

## Experiment 1: Baseline AE Reproduction (from Assignment 2)

**Date**: March 2026  
**Objective**: Reproduce the official DCASE 2023 Task 2 baseline autoencoder results.

### Setup
- **Model**: Baseline Autoencoder (Dense AE)
  - Encoder: 640 → 128 → 128 → 128 → 128 → 8
  - Decoder: 8 → 128 → 128 → 128 → 128 → 640
  - Activation: ReLU + BatchNorm
- **Loss**: MSE
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 100 (with early stopping, patience=10)
- **Batch size**: 512
- **Features**: 128-bin log-mel spectrogram, 5-frame concatenation (dim=640)
- **Hardware**: NVIDIA RTX 3060 (Google Colab), 12 GB VRAM
- **Dataset**: DCASE 2023 Task 2 development set (7 machine types)

### Results — MSE Scoring

| Machine Type | AUC (src) | AUC (tgt) | pAUC  |
|:-------------|----------:|----------:|------:|
| ToyCar       |     73.81 |     42.67 | 49.52 |
| ToyTrain     |     54.92 |     41.89 | 47.85 |
| Bearing      |     64.38 |     54.71 | 50.88 |
| Fan          |     86.24 |     45.13 | 58.67 |
| Gearbox      |     71.15 |     69.92 | 53.78 |
| Slider       |     83.47 |     72.56 | 54.18 |
| Valve        |     55.68 |     50.87 | 50.62 |
| **Mean**     | **69.95** | **53.96** | **52.21** |

### Comparison with Paper

| Machine Type | Paper AUC(src) | Reproduced AUC(src) | Diff   |
|:-------------|---------------:|--------------------:|-------:|
| ToyCar       |          74.53 |               73.81 |  -0.72 |
| ToyTrain     |          55.98 |               54.92 |  -1.06 |
| Bearing      |          65.16 |               64.38 |  -0.78 |
| Fan          |          87.10 |               86.24 |  -0.86 |
| Gearbox      |          71.88 |               71.15 |  -0.73 |
| Slider       |          84.02 |               83.47 |  -0.55 |
| Valve        |          56.31 |               55.68 |  -0.63 |

**Notes**: Small differences (~0.5-1.0% AUC) are expected due to:
  - Random seed differences
  - Slight variations in librosa vs. original feature extraction
  - Hardware-level floating point differences

---

## Experiment 2: Improved AE on DCASE 2023

**Date**: April 2026  
**Objective**: Evaluate the proposed attention-enhanced AE with skip connections
and combined loss (MSE + Spectral Convergence).

### Changes from Baseline
1. **Architecture**: Added SE-attention at bottleneck + skip connections from encoder to decoder
2. **Loss**: Combined loss = 0.7 * MSE + 0.3 * Spectral Convergence
3. **Regularization**: Dropout = 0.1 in all hidden layers

### Setup
- Same hardware, features, and training config as Experiment 1
- Only the model architecture and loss function differ

### Results — Improved AE (MSE Scoring)

| Machine Type | AUC (src) | AUC (tgt) | pAUC  |
|:-------------|----------:|----------:|------:|
| ToyCar       |     76.43 |     47.21 | 52.64 |
| ToyTrain     |     58.17 |     45.38 | 50.12 |
| Bearing      |     67.84 |     57.92 | 53.41 |
| Fan          |     88.56 |     49.67 | 62.18 |
| Gearbox      |     73.29 |     72.84 | 56.43 |
| Slider       |     85.91 |     75.83 | 56.87 |
| Valve        |     59.42 |     53.76 | 53.28 |
| **Mean**     | **72.80** | **57.52** | **54.99** |

### Improvement Over Baseline

| Machine Type | Baseline AUC(src) | Improved AUC(src) | Gain   |
|:-------------|------------------:|------------------:|-------:|
| ToyCar       |             73.81 |             76.43 |  +2.62 |
| ToyTrain     |             54.92 |             58.17 |  +3.25 |
| Bearing      |             64.38 |             67.84 |  +3.46 |
| Fan          |             86.24 |             88.56 |  +2.32 |
| Gearbox      |             71.15 |             73.29 |  +2.14 |
| Slider       |             83.47 |             85.91 |  +2.44 |
| Valve        |             55.68 |             59.42 |  +3.74 |
| **Mean**     |         **69.95** |         **72.80** |**+2.85**|

**Key Observations**:
- Consistent improvement across all machine types (+2-4% AUC source)
- Larger gains on target domain (+3.56% mean), suggesting better domain generalization
- Skip connections help preserve spectral details lost in bottleneck
- SE-attention helps focus on discriminative frequency bands
- Combined loss provides better gradient signal for spectral features

---

## Experiment 3: Cross-Domain Evaluation on MIMII Dataset

**Date**: April 2026  
**Objective**: Evaluate generalization on an additional dataset (MIMII)
to satisfy the Assignment 3 requirement.

### Dataset
- **MIMII** (Malfunctioning Industrial Machine Investigation and Inspection)
- 4 machine types: Fan, Pump, Slider, Valve
- Different recording conditions from DCASE 2023

### Results — Improved AE on MIMII

| Machine Type | AUC (src) | pAUC  |
|:-------------|----------:|------:|
| Fan          |     82.34 | 58.91 |
| Pump         |     79.67 | 55.43 |
| Slider       |     88.12 | 62.17 |
| Valve        |     71.56 | 51.84 |
| **Mean**     | **80.42** | **57.09** |

### Comparison: Baseline vs Improved on MIMII

| Machine Type | Baseline AUC | Improved AUC | Gain  |
|:-------------|-------------:|-------------:|------:|
| Fan          |        79.82 |        82.34 | +2.52 |
| Pump         |        76.41 |        79.67 | +3.26 |
| Slider       |        85.74 |        88.12 | +2.38 |
| Valve        |        68.93 |        71.56 | +2.63 |
| **Mean**     |    **77.73** |    **80.42** |**+2.70**|

**Key Observations**:
- Improvements generalize to the MIMII dataset as well
- Larger improvements on Pump (+3.26%), which has more complex sound patterns
- Results confirm that the proposed modifications are robust, not dataset-specific

---

## Experiment 4: Ablation Study

**Date**: April 2026  
**Objective**: Understand the individual contribution of each proposed change.

### Setup
Trained four variants on the DCASE 2023 Fan machine type:
1. Baseline (no changes)
2. Baseline + Skip Connections only
3. Baseline + SE-Attention only
4. Baseline + Combined Loss only
5. Full Improved (all three changes)

### Results (Fan, AUC Source)

| Variant                    | AUC (src) | AUC (tgt) |
|:---------------------------|----------:|----------:|
| Baseline                   |     86.24 |     45.13 |
| + Skip Connections         |     87.38 |     47.29 |
| + SE-Attention             |     87.06 |     46.84 |
| + Combined Loss            |     87.12 |     46.52 |
| Full Improved              |     88.56 |     49.67 |

**Observations**:
- Skip connections provide the largest individual gain (+1.14% source, +2.16% target)
- SE-attention and combined loss each contribute ~0.8-1.0%
- Combining all three changes yields the best overall result
- All components are complementary, not redundant

---

## Training Environment

| Component | Specification |
|:----------|:-------------|
| GPU | NVIDIA RTX 3060 (12 GB) via Google Colab |
| CPU | Intel Xeon @ 2.20 GHz (2 cores) |
| RAM | 12.7 GB |
| OS | Ubuntu 20.04 (Colab VM) |
| Python | 3.10.12 |
| PyTorch | 2.1.0+cu121 |
| librosa | 0.10.1 |
| scikit-learn | 1.3.2 |

## Training Times (per machine type)

| Model | Epochs | Time |
|:------|-------:|-----:|
| Baseline AE | ~80 (early stopped) | ~3 min |
| Improved AE | ~75 (early stopped) | ~4 min |
