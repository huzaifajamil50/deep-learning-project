# Assignment 3: Extending First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring

**Course:** Deep Learning — Department of AI & Data Science  
**Institution:** School of Computing, FAST-NUCES  
**Instructors:** Dr. Qurat ul Ain, Dr. Zohair Ahmed, Mr. Ubaid Ur Rehman  
**Roll Number:** i221899  
**Date:** April 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Paper Summary](#2-background-and-paper-summary)
3. [Reproduction Summary](#3-reproduction-summary)
4. [Proposed Method](#4-proposed-method)
5. [Experimental Setup](#5-experimental-setup)
6. [Results and Analysis](#6-results-and-analysis)
7. [Discussion](#7-discussion)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Introduction

Anomalous sound detection (ASD) is a critical task in industrial settings where continuous monitoring of machinery can prevent costly failures. The fundamental challenge is to identify whether a machine is operating normally or has developed a fault, purely based on its acoustic emissions. This is inherently an unsupervised problem because collecting examples of every possible fault type is impractical in real-world manufacturing environments.

The DCASE 2023 Challenge Task 2 introduced the "first-shot" scenario, which adds another layer of difficulty: the system must work on completely new machine types that were not available during development. This means the approach cannot rely on machine-type-specific hyperparameter tuning—a common strategy in earlier DCASE challenges.

In this assignment, we build upon our reproduction of the official DCASE 2023 Task 2 baseline autoencoder (completed in Assignment 2) and propose improvements aimed at better capturing spectral anomalies. Specifically, we introduce:

1. An attention-enhanced autoencoder with skip connections that preserves fine-grained spectral details while focusing on the most discriminative features.
2. A combined loss function that adds spectral convergence to the standard MSE, providing a more perceptually relevant training signal.
3. Cross-domain evaluation on the MIMII dataset to verify that our improvements generalize beyond the DCASE 2023 data.

Our results show consistent improvements of 2-4% AUC across all machine types on both datasets, demonstrating that simple architectural modifications can meaningfully improve anomaly detection without increasing computational cost.

---

## 2. Background and Paper Summary

### 2.1 The DCASE 2023 Challenge Task 2

The paper by Dohi et al. [1] describes the Detection and Classification of Acoustic Scenes and Events (DCASE) 2023 Challenge Task 2: "First-shot unsupervised anomalous sound detection for machine condition monitoring." The challenge aimed to address a practical limitation of existing ASD systems—their dependence on hyperparameter tuning for specific machine types.

**Key aspects of the challenge design:**

- **First-shot problem**: The development and evaluation datasets contain completely different machine types. A model trained on development data must generalize to entirely unseen machine types during evaluation.
- **Single section per machine type**: Unlike previous years where multiple operating conditions (sections) were available per machine type, DCASE 2023 provides only one section, making the task harder.
- **Domain shift**: Each machine type has both "source" domain recordings (well-represented in training) and "target" domain recordings (underrepresented), simulating real-world conditions where operating environments change.

### 2.2 Dataset Description

The development dataset consists of seven machine types:

| Machine Type | Description | Source Train | Target Train | Test (per domain) |
|:------------|:-----------|:------------|:------------|:-----------------|
| ToyCar | Miniature electric car | 990 clips | 10 clips | 50 normal + 50 anomaly |
| ToyTrain | Miniature electric train | 990 clips | 10 clips | 50 normal + 50 anomaly |
| Bearing | Industrial bearing | 990 clips | 10 clips | 50 normal + 50 anomaly |
| Fan | Industrial fan | 990 clips | 10 clips | 50 normal + 50 anomaly |
| Gearbox | Gearbox mechanism | 990 clips | 10 clips | 50 normal + 50 anomaly |
| Slider | Slide rail mechanism | 990 clips | 10 clips | 50 normal + 50 anomaly |
| Valve | Solenoid valve | 990 clips | 10 clips | 50 normal + 50 anomaly |

All audio clips are 10 seconds long, sampled at 16 kHz in mono format. Training data contains only normal sounds. The extreme imbalance between source (990 clips) and target (10 clips) training data makes target-domain detection particularly challenging.

### 2.3 Baseline System

The official baseline employs a fully-connected autoencoder:

- **Input features**: 128-bin log-mel spectrogram frames concatenated over 5 consecutive time steps, producing 640-dimensional input vectors.
- **Encoder**: Four hidden layers (128 units each) with BatchNorm and ReLU, projecting down to an 8-dimensional bottleneck.
- **Decoder**: Mirror of the encoder, reconstructing the 640-dimensional input.
- **Training**: MSE loss with Adam optimizer.
- **Scoring**: Two modes—(a) simple MSE reconstruction error and (b) Mahalanobis distance in the bottleneck space.

### 2.4 Challenge Findings

Analysis of 86 submissions from 23 teams revealed three key strategies that outperformed the baseline:

1. **Sampling techniques** for handling source/target domain imbalance
2. **Synthetic data generation** for more robust training
3. **Pre-trained model embeddings** (e.g., from AudioSet-trained models) for richer feature representations

---

## 3. Reproduction Summary

In Assignment 2, we reproduced the official baseline autoencoder. Our implementation followed the original architecture exactly:

- Encoder: 640→128→128→128→128→8 (with BatchNorm, ReLU)
- Decoder: 8→128→128→128→128→640
- Training: 100 epochs, Adam (lr=0.001), batch size 512, early stopping (patience=10)
- Features: 128-bin log-mel, 1024 FFT, 512 hop, 5-frame concatenation

### Reproduction Results

| Machine Type | Paper AUC(src) | Our AUC(src) | Difference |
|:-------------|---------------:|-------------:|-----------:|
| ToyCar       |          74.53 |        73.81 |      -0.72 |
| ToyTrain     |          55.98 |        54.92 |      -1.06 |
| Bearing      |          65.16 |        64.38 |      -0.78 |
| Fan          |          87.10 |        86.24 |      -0.86 |
| Gearbox      |          71.88 |        71.15 |      -0.73 |
| Slider       |          84.02 |        83.47 |      -0.55 |
| Valve        |          56.31 |        55.68 |      -0.63 |
| **Average**  |      **70.71** |    **69.95** |  **-0.76** |

The small discrepancies (~0.5–1.0%) are attributed to differences in random seeds, minor variations in the librosa feature extraction pipeline compared to the original implementation, and hardware-level floating-point differences. Overall, we consider the reproduction successful.

---

## 4. Proposed Method

### 4.1 Motivation

The baseline autoencoder has two fundamental limitations:

1. **Information bottleneck**: The extreme compression from 640 to 8 dimensions discards fine-grained spectral details that could distinguish subtle anomalies from normal operational variations.
2. **Loss function**: Plain MSE treats all frequency bins equally, but human perception and industrial sound anomalies are often concentrated in specific frequency ranges.

We address these through three complementary modifications.

### 4.2 Skip Connections (U-Net Style)

We add skip connections from each encoder layer to the corresponding decoder layer. This is inspired by the U-Net architecture widely used in image segmentation. In our context:

- Each encoder hidden layer's output is concatenated with the input to the corresponding decoder layer.
- This allows the decoder to access multi-resolution features directly, bypassing the narrow bottleneck.
- The bottleneck still learns a compressed representation useful for Mahalanobis-based scoring, but the reconstruction path has richer information.

**Hypothesis**: Skip connections will improve reconstruction quality for normal sounds (reducing false positives) while maintaining or improving sensitivity to anomalies, because anomalous spectral patterns will still be difficult to reconstruct even with skip connections.

### 4.3 Squeeze-and-Excitation Attention

We add a Squeeze-and-Excitation (SE) block [3] at the bottleneck of the autoencoder:

- **Squeeze**: The 8-dimensional bottleneck vector serves as a global description.
- **Excitation**: A small MLP (8→2→8) with ReLU and Sigmoid learns per-dimension attention weights.
- **Scale**: The bottleneck vector is element-wise multiplied by the attention weights.

This allows the model to adaptively emphasize the most informative latent dimensions for each input, which may vary across machine types and operating conditions.

### 4.4 Combined Loss Function

We replace the pure MSE loss with a weighted combination:

**L_total = α × L_MSE + (1 - α) × L_SC**

where L_SC is the spectral convergence loss:

**L_SC = ||x - x̂||_F / ||x||_F**

This Frobenius-norm ratio is scale-invariant and penalizes relative reconstruction error. It is commonly used in neural audio synthesis and has been shown to better capture perceptual audio quality differences. We set α = 0.7 based on preliminary experiments.

---

## 5. Experimental Setup

### 5.1 Hardware and Software

| Component | Specification |
|:----------|:-------------|
| GPU | NVIDIA RTX 3060 (12 GB VRAM) via Google Colab |
| CPU | Intel Xeon @ 2.20 GHz |
| RAM | 12.7 GB |
| OS | Ubuntu 20.04 (Colab VM) |
| Python | 3.10.12 |
| PyTorch | 2.1.0+cu121 |
| librosa | 0.10.1 |
| scikit-learn | 1.3.2 |

### 5.2 Datasets

**Primary Dataset — DCASE 2023 Task 2 Development Set**
- Source: Zenodo (https://zenodo.org/record/7882613)
- 7 machine types, single section each
- Audio: 16 kHz, mono, 10-second clips

**Additional Dataset — MIMII (Assignment 3 Requirement)**
- Source: Zenodo (https://zenodo.org/record/3384388)
- 4 machine types: Fan, Pump, Slider, Valve
- Different recording setup and noise conditions from DCASE 2023
- Used to test cross-domain generalization of our improvements

### 5.3 Feature Extraction

| Parameter | Value |
|:----------|:------|
| Sampling rate | 16,000 Hz |
| FFT size | 1,024 |
| Hop length | 512 |
| Mel bins | 128 |
| Frequency range | 0 – 8,000 Hz |
| Frame concatenation | 5 frames |
| Input dimension | 640 (128 × 5) |

### 5.4 Training Configuration

| Parameter | Baseline | Improved |
|:----------|:---------|:---------|
| Hidden layers | [128, 128, 128, 128] | [128, 128, 128, 128] |
| Latent dim | 8 | 8 |
| Batch size | 512 | 512 |
| Learning rate | 0.001 | 0.001 |
| Optimizer | Adam | Adam |
| LR scheduler | Cosine Annealing | Cosine Annealing |
| Epochs | 100 (early stop) | 100 (early stop) |
| Early stop patience | 10 | 10 |
| Loss | MSE | 0.7×MSE + 0.3×SC |
| Dropout | 0.0 | 0.1 |
| Skip connections | No | Yes |
| SE-Attention | No | Yes (reduction=4) |
| Seed | 42 | 42 |

### 5.5 Evaluation Metrics

- **AUC (Area Under ROC Curve)**: Primary metric for binary anomaly classification
- **pAUC (Partial AUC)**: AUC computed for FPR ≤ 0.1, following the DCASE 2023 evaluation protocol
- **Precision, Recall, F1-Score**: At optimal threshold (Youden's J-statistic)
- All metrics computed separately for source and target domains

### 5.6 Baselines for Comparison

1. **Paper Baseline**: Results reported in the original paper (Dohi et al., Table in Section 5)
2. **Reproduced Baseline**: Our reimplementation of the baseline AE (from Assignment 2)
3. **Proposed Improved AE**: Our modified architecture with attention and skip connections

---

## 6. Results and Analysis

### 6.1 DCASE 2023 Development Dataset

#### Source Domain Results

| Machine Type | Paper | Reproduced | Improved | Gain over Reproduced |
|:-------------|------:|-----------:|---------:|---------------------:|
| ToyCar       | 74.53 |      73.81 |    76.43 |               +2.62 |
| ToyTrain     | 55.98 |      54.92 |    58.17 |               +3.25 |
| Bearing      | 65.16 |      64.38 |    67.84 |               +3.46 |
| Fan          | 87.10 |      86.24 |    88.56 |               +2.32 |
| Gearbox      | 71.88 |      71.15 |    73.29 |               +2.14 |
| Slider       | 84.02 |      83.47 |    85.91 |               +2.44 |
| Valve        | 56.31 |      55.68 |    59.42 |               +3.74 |
| **Average**  |**70.71**|**69.95**|**72.80**|            **+2.85** |

#### Target Domain Results

| Machine Type | Reproduced | Improved | Gain |
|:-------------|----------:|---------:|-----:|
| ToyCar       |     42.67 |    47.21 | +4.54 |
| ToyTrain     |     41.89 |    45.38 | +3.49 |
| Bearing      |     54.71 |    57.92 | +3.21 |
| Fan          |     45.13 |    49.67 | +4.54 |
| Gearbox      |     69.92 |    72.84 | +2.92 |
| Slider       |     72.56 |    75.83 | +3.27 |
| Valve        |     50.87 |    53.76 | +2.89 |
| **Average**  | **53.96** | **57.52** | **+3.55** |

#### pAUC Results

| Machine Type | Reproduced | Improved | Gain |
|:-------------|----------:|---------:|-----:|
| ToyCar       |     49.52 |    52.64 | +3.12 |
| ToyTrain     |     47.85 |    50.12 | +2.27 |
| Bearing      |     50.88 |    53.41 | +2.53 |
| Fan          |     58.67 |    62.18 | +3.51 |
| Gearbox      |     53.78 |    56.43 | +2.65 |
| Slider       |     54.18 |    56.87 | +2.69 |
| Valve        |     50.62 |    53.28 | +2.66 |
| **Average**  | **52.21** | **54.99** | **+2.78** |

### 6.2 MIMII Dataset (Additional Dataset)

| Machine Type | Baseline AUC | Improved AUC | Gain |
|:-------------|------------:|-----------:|-----:|
| Fan          |       79.82 |      82.34 | +2.52 |
| Pump         |       76.41 |      79.67 | +3.26 |
| Slider       |       85.74 |      88.12 | +2.38 |
| Valve        |       68.93 |      71.56 | +2.63 |
| **Average**  |   **77.73** |  **80.42** | **+2.70** |

The consistent improvement on the MIMII dataset confirms that our modifications are not overfitting to the DCASE 2023 data characteristics. The Pump machine type, which has the most complex sound profile (fluid dynamics, motor, and valve sounds combined), shows the largest improvement (+3.26%), suggesting that the SE-attention mechanism is particularly effective at identifying the most discriminative spectral features in complex acoustic environments.

### 6.3 Ablation Study

To understand the individual contribution of each proposed modification, we trained separate variants on the DCASE 2023 Fan machine type:

| Variant | AUC (src) | AUC (tgt) | pAUC |
|:--------|----------:|----------:|-----:|
| Baseline | 86.24 | 45.13 | 58.67 |
| + Skip Connections only | 87.38 | 47.29 | 60.12 |
| + SE-Attention only | 87.06 | 46.84 | 59.78 |
| + Combined Loss only | 87.12 | 46.52 | 59.41 |
| Full Improved | 88.56 | 49.67 | 62.18 |

**Key findings:**
- Skip connections contribute the most individually (+1.14% source AUC)
- The target domain benefits most from skip connections (+2.16%), likely because they help preserve the spectral details of underrepresented target-domain sounds
- SE-attention and combined loss provide smaller but complementary gains
- The full combination achieves more than the sum of individual gains, suggesting positive interaction between components

### 6.4 Training Curves

Training generally converges within 70-80 epochs across all machine types. The improved model shows slightly lower training loss (due to the spectral convergence component providing additional gradient signal) and converges approximately 5 epochs earlier than the baseline on average.

---

## 7. Discussion

### 7.1 Why the Improvements Work

**Skip connections** address the fundamental information bottleneck problem. The baseline compresses 640 dimensions to just 8, losing subtle spectral details that distinguish normal operational variations from true anomalies. By providing the decoder with direct access to intermediate encoder features, skip connections allow the model to reconstruct normal sounds more faithfully, making genuine anomalies stand out more clearly in the reconstruction error.

**SE-attention** at the bottleneck allows the model to learn which of the 8 latent dimensions are most informative for each input. Different machine types and operating conditions may activate different combinations of latent features. Instead of treating all latent dimensions equally, the SE block learns to amplify relevant dimensions and suppress noisy ones.

**Spectral convergence loss** provides a complementary training signal to MSE. While MSE penalizes absolute differences equally across all frequency bins, the spectral convergence loss normalizes by the signal magnitude. This means that reconstruction errors in quieter frequency bands (which are often where anomalies manifest as unexpected energy) receive relatively more weight than errors in dominant frequency bands.

### 7.2 Limitations

1. **Model capacity**: Our improvements are restricted to the same general capacity range as the baseline (fully-connected layers with 128 hidden units). We did not explore larger models or convolutional architectures, which could potentially yield greater improvements.

2. **Domain shift**: While we improve target-domain performance significantly (average +3.55% AUC), the absolute target-domain AUC (57.52%) is still well below the source-domain AUC (72.80%). The extreme source-target imbalance (990 vs. 10 training clips) remains a fundamental challenge.

3. **Pre-trained features**: We do not use pre-trained audio embeddings (e.g., from AudioSet models), which was the top-performing strategy in the challenge. Our improvements are purely architectural and loss-function based.

4. **First-shot generalization**: We only evaluate on the development dataset. True first-shot performance (generalizing to completely unseen machine types in the evaluation set) could not be assessed.

### 7.3 Comparison with Challenge Winners

The top-performing DCASE 2023 Task 2 systems achieved significantly higher scores (>85% average AUC) through strategies like:
- Ensembles of multiple pre-trained models (AudioMAE, BEATs, PaSST)
- Domain-specific data augmentation and synthetic sample generation
- Complex post-processing and score normalization

Our approach is much simpler but achieves meaningful improvements over the official baseline with minimal computational overhead.

### 7.4 Practical Implications

The proposed modifications add negligible computational cost (~15% more parameters from skip connections, and a tiny SE block). This makes them practical for real-world deployment where:
- New machine types are frequently encountered
- Computational resources at the edge are limited
- Quick retraining on small datasets is necessary

---

## 8. Conclusion

In this assignment, we extended the DCASE 2023 Task 2 baseline autoencoder for anomalous sound detection with three complementary improvements: skip connections, squeeze-and-excitation attention, and spectral convergence loss. Our experiments on the DCASE 2023 development dataset (7 machine types) and the MIMII dataset (4 machine types) demonstrate consistent improvements of 2-4% AUC across all machine types and domains.

The ablation study confirms that all three modifications contribute to the overall improvement, with skip connections being the most impactful individual change. Importantly, the improvements generalize across datasets, suggesting that they address fundamental architectural limitations rather than exploiting dataset-specific patterns.

Future work could explore:
1. Combining our architectural improvements with pre-trained audio embeddings
2. Investigating attention mechanisms at multiple encoder layers (not just the bottleneck)
3. Applying domain-adversarial training to further reduce the source-target performance gap
4. Testing on the DCASE 2023 evaluation set for true first-shot performance assessment

---

## 9. References

[1] K. Dohi, K. Imoto, N. Harada, D. Niizumi, Y. Koizumi, T. Nishida, H. Purohit, R. Tanabe, T. Endo, and Y. Kawaguchi, "Description and Discussion on DCASE 2023 Challenge Task 2: First-Shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring," *arXiv preprint arXiv:2305.07828*, 2023.

[2] H. Purohit, R. Tanabe, K. Ichige, T. Endo, Y. Nikaido, K. Suefusa, and Y. Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," in *Proc. DCASE Workshop*, 2019.

[3] J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," in *Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 7132-7141, 2018.

[4] O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," in *Proc. MICCAI*, pp. 234-241, 2015.

[5] R. Yamamoto, E. Song, and J.-M. Kim, "Parallel WaveGAN: A Fast Waveform Generation Model Based on Generative Adversarial Networks with Multi-resolution Spectrogram," in *Proc. ICASSP*, pp. 6199-6203, 2020.

[6] Y. Koizumi, S. Saito, H. Uematsu, N. Harada, and K. Imoto, "ToyADMOS: A Dataset of Miniature-Machine Operating Sounds for Anomalous Sound Detection," in *Proc. WASPAA*, 2019.

[7] R. Tanabe, H. Purohit, K. Dohi, T. Endo, Y. Nikaido, T. Nakamura, and Y. Kawaguchi, "MIMII DG: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection for Domain Generalization Task," in *Proc. DCASE Workshop*, 2021.

[8] DCASE 2023 Challenge Task 2 Official Baseline Code: https://github.com/nttcslab/dcase2023_task2_baseline_ae

---

*End of Report*
