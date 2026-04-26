# DCASE 2023 Task 2 — Anomalous Sound Detection for Machine Condition Monitoring

## Extending First-Shot Unsupervised ASD with Attention-Enhanced Autoencoders

> **Course**: Deep Learning (DS/AI) — FAST-NUCES  
> **Assignment 3**: Original Research Extension  
> **Base Paper**: Dohi et al., *"Description and Discussion on DCASE 2023 Challenge Task 2"*, [arXiv:2305.07828](https://arxiv.org/abs/2305.07828)

---

## Project Overview

This repository contains our implementation and extension of the DCASE 2023 Challenge Task 2 baseline system for anomalous sound detection (ASD) in machine condition monitoring. The goal is to detect whether a machine is operating normally or abnormally based on its acoustic emissions, without having seen anomalous examples during training.

### What We Did

1. **Reproduced** the official DCASE 2023 baseline autoencoder (Assignment 2)
2. **Proposed improvements** with an attention-enhanced AE featuring skip connections and a combined spectral convergence loss (Assignment 3)
3. **Evaluated** on an additional dataset (MIMII) for cross-domain generalization

### Key Results

| Model | DCASE 2023 Avg AUC (src) | MIMII Avg AUC | Improvement |
|:------|---:|---:|---:|
| Baseline AE | 69.95% | 77.73% | — |
| Improved AE | 72.80% | 80.42% | +2.85% / +2.70% |

---

## Repository Structure

```
├── configs/
│   └── config.yaml              # All hyperparameters and paths
├── src/
│   ├── __init__.py
│   ├── dataset.py               # Data loading and preprocessing
│   ├── feature_extraction.py    # Log-mel spectrogram extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── baseline_ae.py       # Baseline Autoencoder
│   │   └── improved_ae.py       # Proposed: AE + Attention + Skip Connections
│   ├── losses.py                # MSE + Spectral Convergence Loss
│   ├── train.py                 # Training loop
│   ├── test.py                  # Evaluation pipeline
│   └── utils.py                 # Metrics, plotting, helpers
├── scripts/
│   ├── download_data.py         # Dataset download helper
│   ├── run_baseline.py          # Run baseline experiments
│   └── run_improved.py          # Run improved model experiments
├── notebooks/
│   └── experiments.ipynb        # Google Colab notebook
├── results/                     # Output CSV files and figures
├── logs/
│   └── experiment_logs.md       # Detailed experiment documentation
├── report/
│   └── Assignment3_Report.md    # Full assignment report
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/huzaifajamil50/deep-learning-project.git
cd deep-learning-project
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
# Download DCASE 2023 Task 2 development dataset
python scripts/download_data.py --dataset dcase2023

# Download MIMII dataset (additional dataset for Assignment 3)
python scripts/download_data.py --dataset mimii
```

### 3. Run Baseline Experiments

```bash
python scripts/run_baseline.py
```

### 4. Run Improved Model Experiments

```bash
python scripts/run_improved.py
```

### 5. Or Use Google Colab

Open `notebooks/experiments.ipynb` in Google Colab for a complete walkthrough.

---

## Methodology

### Baseline Autoencoder

The baseline follows the official DCASE 2023 Task 2 system:
- **Input**: 640-dim vectors (128 mel bins × 5 frames)
- **Encoder**: 640 → 128 → 128 → 128 → 128 → 8
- **Decoder**: 8 → 128 → 128 → 128 → 128 → 640
- **Loss**: MSE reconstruction error
- **Scoring**: Average MSE per test clip

### Proposed Improvements

1. **SE-Attention Block** at the bottleneck: Adaptively reweights latent dimensions
2. **Skip Connections**: Forward encoder hidden states to decoder (U-Net style)
3. **Combined Loss**: 0.7×MSE + 0.3×Spectral Convergence

### Datasets

| Dataset | Machine Types | Year |
|:--------|:-------------|:-----|
| DCASE 2023 Task 2 | ToyCar, ToyTrain, Bearing, Fan, Gearbox, Slider, Valve | 2023 |
| MIMII | Fan, Pump, Slider, Valve | 2019 |

### Evaluation Metrics

- **AUC**: Area Under ROC Curve
- **pAUC**: Partial AUC (max FPR = 0.1)
- Evaluated separately for source and target domains

---

## Hardware

| Component | Specification |
|:----------|:-------------|
| GPU | NVIDIA RTX 3060 (12 GB) via Google Colab |
| Python | 3.10 |
| PyTorch | 2.1.0 |

---

## References

1. Dohi, K., et al. "Description and Discussion on DCASE 2023 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring." arXiv:2305.07828, 2023.
2. Purohit, H., et al. "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection." DCASE Workshop, 2019.
3. Hu, J., Shen, L., & Sun, G. "Squeeze-and-Excitation Networks." CVPR, 2018.
4. Official Baseline Code: https://github.com/nttcslab/dcase2023_task2_baseline_ae

---

## Team

- Roll Number: i221899
- Course: Deep Learning — Department of AI & Data Science, FAST-NUCES
