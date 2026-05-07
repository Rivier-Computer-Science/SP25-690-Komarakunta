# Robust Traffic Sign Recognition under Real-World Visual Degradation

**A Comparative Study of CNN, ResNet, and Transformer Architectures**

Final Project for COMP-690: Deep Learning (Spring 2026)  
Author: Ujwal Komarakunta, Rivier University

---

## Overview

Traffic sign recognition is a safety-critical task in autonomous driving. While deep learning models achieve high accuracy on clean benchmark datasets, real-world conditions introduce visual degradations—blur, noise, fog, and lighting changes—that can cause dangerous misclassifications. This project systematically compares four architectures under synthetic visual degradation to evaluate real-world robustness beyond clean-image accuracy.

---

## Research Questions

1. How do different architectures (traditional vs deep vs transformer) compare in robustness under real-world visual degradation?
2. Does training with degradation augmentation improve robustness without sacrificing clean-image accuracy?
3. Which architecture is most suitable for safety-critical deployment under specific degradation types?

---

## Dataset

**German Traffic Sign Recognition Benchmark (GTSRB)**
- 39,209 training images (split: 23,976 train / 2,664 validation)
- 12,630 test images
- 43 traffic sign categories
- Source: [GTSRB on Kaggle](https://www.kaggle.com/datasets/meowmeowmeow/gtsrb-german-traffic-sign)

**Synthetic Degradations Applied:**
- Gaussian Blur (kernel sizes: 3, 5, 7)
- Additive Noise (std: 5, 10, 20)
- Low Brightness (factors: 0.8, 0.6, 0.4, 0.2)
- Fog (intensities: 0.2, 0.35, 0.5, 0.65)

---

## Models Compared

1. **HOG + SVM** — Traditional computer vision baseline (hand-crafted features)
2. **Simple CNN** — Lightweight 3-layer CNN trained from scratch
3. **ResNet-18** — Pretrained on ImageNet, fine-tuned on GTSRB
4. **Vision Transformer (ViT-Base)** — Pretrained on ImageNet-21k, fine-tuned on GTSRB

---

## Key Results

### Clean Image Accuracy
| Model | Test Accuracy | Macro F1 |
|-------|--------------|----------|
| HOG+SVM | 87.17% | 0.8555 |
| Simple CNN | 93.65% | 0.8808 |
| ResNet-18 | 96.58% | 0.9206 |
| ViT-Base | 95.61% | 0.9028 |

### Robustness Findings
- **Blur**: ViT most robust (82.39% at severity 3)
- **Noise**: Simple CNN most robust (86.26%); ResNet drops 32.68 points
- **Brightness**: ResNet maintains 91.62% even at severity 4
- **Fog**: ViT dominates at 93.37%, outperforming ResNet by 15.60 points

### Degradation Augmentation Impact
**ResNet-18:**
- Fog S4: +22.64% improvement (72.64% → 95.27%)
- Noise S3: +13.38% improvement
- Clean accuracy: +0.57% (no tradeoff!)

**ViT:**
- Brightness S4: +14.53% improvement (74.20% → 88.73%)
- Fog S4: +7.54% improvement
- Clean accuracy: +0.68% (no tradeoff!)

---

## Repository Structure
├── notebooks/
│   ├── 01_data_exploration.ipynb.    # Dataset analysis and visualization
│   ├── 02_baseline_models.ipynb      # HOG+SVM baseline
│   ├── 03_simple_cnn.ipynb           # Simple CNN training
│   ├── 04_resnet18.ipynb             # ResNet-18 fine-tuning
│   ├── 05_vit.ipynb                  # Vision Transformer fine-tuning
│   ├── 06_robustness_evaluation.ipynb # Degradation experiments
│   ├── 07_ablation_study.ipynb       # ResNet augmentation ablation
│   └── 08_vit_ablation.ipynb          # ViT augmentation ablation
├── results/figures/                   # Saved plots and visualizations
├── CHANGELOG.md                      # Complete development history
└── README.md                          #This file
---

## Setup & Execution

### Environment
All experiments were conducted on Kaggle with GPU acceleration (T4 x2).

### Dependencies
```bash
pip install torch torchvision transformers opencv-python numpy pandas matplotlib scikit-learn

Running the Notebooks
Upload notebooks to Kaggle or Google Colab
Enable GPU accelerator
Load GTSRB dataset from Kaggle Datasets
Execute notebooks in order (01 → 08)

Key Findings
Clean accuracy is a poor predictor of real-world robustness
Models scored 93-97% on clean data but showed 15-32 point gaps under degradation.
Architecture choice matters for specific degradation types
ViT: Best for fog and blur (global attention advantage)
Simple CNN: Best for noise (trained on noisy data from scratch)
ResNet: Best for brightness (batch normalization handles intensity shifts)
Pretraining on clean data can hurt noise robustness
ResNet (pretrained on pristine ImageNet) showed worst noise performance, dropping 32 points.
Degradation augmentation provides free robustness
Improvements of 13-22% under degradation with zero clean accuracy cost.
Transformers excel under global degradation
ViT outperformed CNNs by 15 points under fog where local features disappear.

Limitations
Image size inconsistency (64×64 vs 224×224) across models
Dataset limited to European traffic signs (geographic bias)
Degradations tested independently, not in combination
Synthetic degradations may not fully capture real-world complexity

Future Work
Evaluate all models at consistent resolution
Test combined degradations (e.g., fog + noise simultaneously)
Extend to geographically diverse traffic sign datasets
Deploy models on real driving footage for validation

Acknowledgments
AI Assistance: Claude (Anthropic) was used for code implementation assistance, debugging, and structure guidance. All code was reviewed, tested, and adapted by the author.
Pretrained Models:
ResNet-18: torchvision.models.resnet18(pretrained=True)
ViT-Base: google/vit-base-patch16-224 (HuggingFace)
Dataset: German Traffic Sign Recognition Benchmark (Stallkamp et al., 2012)

Repository: https://github.com/Rivier-Computer-Science/SP25-690-Komarakunta
