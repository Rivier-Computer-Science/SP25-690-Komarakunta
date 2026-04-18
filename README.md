# Robust Traffic Sign Recognition under Real-World Visual Degradation
### A Comparative Study of CNN, ResNet, and Transformer Architectures

## Proposal

### Problem and Motivation
Recognising traffic signs is an important safety task for self-driving cars. Deep learning 
models work well on clean benchmark data, but in the real world, things like blur, noise, 
fog, and bad lighting can often make images worse, which can lead to dangerous 
misclassifications. People don't often check this difference between benchmark accuracy 
and real-world reliability. This project fills that gap by comparing three architectures 
— a simple CNN, ResNet-18, and Vision Transformer — on how well they can classify traffic 
signs when the images are degraded. This is a meaningful comparison beyond just clean-image 
accuracy because CNNs and ResNets use local convolution and ViT uses global self-attention. 
The goal is to understand which architecture is most robust under degradation and whether 
augmentation training helps close the performance gap.

### Dataset
German Traffic Sign Recognition Benchmark (GTSRB)
- 39,209 total training images, split into 23,976 training and 2,664 validation samples
- 12,630 test samples
- 43 traffic sign categories
- Publicly available on Kaggle
- Synthetic degradations applied: Gaussian blur, additive noise, brightness reduction, fog

### Models
1. HOG + SVM — non-deep traditional baseline
2. Simple CNN — neural baseline, trained from scratch
3. ResNet-18 — fine-tuned residual network
4. Vision Transformer (ViT-Base) — fine-tuned transformer model

### Experiments
1. Baseline comparison of all models on clean test images
2. Robustness evaluation under increasing degradation severity
3. Ablation: ResNet-18 and ViT trained with vs without degradation augmentation

### Evaluation Metrics
- Top-1 Accuracy (primary)
- Per-class accuracy
- Confusion matrix
- Accuracy vs degradation severity curves
- F1-Score

### Timeline
- Week 1: Data pipeline, preprocessing, HOG+SVM and Simple CNN baseline
- Week 2: ResNet-18 and ViT fine-tuning, clean-image evaluation
- Week 3: Robustness experiments, ablation, failure analysis, report and GitHub

## Project Structure
- `data/` — GTSRB dataset
- `notebooks/` — Colab notebooks
- `src/` — training and evaluation scripts
- `results/` — figures, tables, saved outputs

## Setup
Install dependencies:
pip install -r requirements.txt

## Dataset
Download GTSRB from: https://www.kaggle.com/datasets/meowmeowmeow/gtsrb-german-traffic-sign
Unzip into the data/ folder.

## Acknowledgements
Code assistance provided by Claude (Anthropic). All code was reviewed, 
tested, and adapted by the author. Pretrained models sourced from 
torchvision and HuggingFace.
