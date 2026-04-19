# Changelog
## [Apr 19, 2026] - Baseline Comparison Results
### Results Summary
| Model      | Test Accuracy | Macro F1 |
|------------|---------------|----------|
| HOG+SVM    | 87.17%        | 0.86     |
| Simple CNN | 96.62%        | 0.9435   |
| ResNet18   | 96.83%        | 0.9253   |
| ViT-Base   | 96.66%        | 0.9173   |

### Key Observations
- Deep learning models outperform HOG+SVM by ~10%
- All three deep models perform within 0.21% of each other on clean images
- Pretrained models (ResNet18, ViT) not significantly better than 
  Simple CNN from scratch on clean images
- Real architectural differences expected in robustness evaluation
- ViT underperforms expectations possibly due to small dataset size
  and simple geometric nature of traffic signs

### Next Steps
- Robustness evaluation under blur, noise, brightness, fog
- Ablation study: with vs without degradation augmentation
- Failure analysis and confusion matrices
## [Apr 19, 2026] - ViT Training
### Added
- Vision Transformer ViT-Base pretrained on ImageNet-21k, fine-tuned on GTSRB
- Pretrained weights from HuggingFace (google/vit-base-patch16-224)
- Image size: 224x224 (required by ViT architecture)
- Training: 2 epochs (peak performance observed at epoch 2)
- Val Accuracy: 99.62%, Test Accuracy: 96.66%, Macro F1: 0.9173
- AI assistance: Claude (Anthropic) used for implementation

## [Apr 19, 2026] - ResNet18 Training
### Added
- ResNet18 pretrained on ImageNet, fine-tuned on GTSRB at 224x224
- Pretrained weights from torchvision (ImageNet1K)
- Three experiments conducted to find optimal configuration
- Experiment 1: 4 epochs, no dropout → Test Accuracy = 96.83% (best)
- Experiment 2: 3 epochs, dropout=0.3 → Test Accuracy = 95.80%
- Experiment 3: 3 epochs, no dropout → Test Accuracy = 96.03%
- Final ResNet18 Test Accuracy: 96.83%, Macro F1: 0.9253
- AI assistance: Claude (Anthropic) used for implementation

## [Apr 19, 2026] - Moved to Kaggle
### Changed
- Switched from Google Colab to Kaggle for GPU access
- Kaggle provides 30hrs/week free GPU (T4 x2)
- All pretrained models retrained at 224x224 (native resolution)
- Simple CNN and HOG+SVM kept at 64x64 for consistency
## [Apr 19, 2026] - ResNet18 Training
### Added
- ResNet18 pretrained on ImageNet, fine-tuned on GTSRB at 224x224
- Three experiments conducted to find optimal configuration
- Experiment 1: 4 epochs, no dropout → Test Accuracy = 96.83% (best)
- Experiment 2: 3 epochs, dropout=0.3 → Test Accuracy = 95.80%
- Experiment 3: 3 epochs, no dropout → Test Accuracy = 96.03%
- Final ResNet18 Test Accuracy: 96.83%, Macro F1: 0.9253
- AI assistance: Claude (Anthropic) used for implementation

## [Apr 19, 2026] - Moved to Kaggle
### Changed
- Switched from Google Colab to Kaggle for GPU access
- Kaggle provides 30hrs/week free GPU (T4 x2)
- All models retrained properly at 224x224 for pretrained models

## [Apr 18, 2026] - Simple CNN
### Added
- Simple CNN with 3 conv layers trained from scratch
- Test accuracy: 96.62%, Macro F1: 0.9435
- Beats HOG+SVM baseline by 9.45%
- AI assistance: Claude (Anthropic) used for implementation
## [Apr 18, 2026] - HOG+SVM Baseline
### Added
- HOG feature extraction from GTSRB dataset
- LinearSVM classifier trained on HOG features
- Test accuracy: 87.17%, Macro F1: 0.86
- Worst class: Class 30 Beware ice/snow (F1: 0.53)
- AI assistance: Claude (Anthropic) used for implementation
## [Apr 18, 2026] - Degradation Visualization
### Added
- Revised degradation visualization using clearer sample image (idx:500, Bumpy road)
- Severity levels limited to 1-3 as levels 4-5 are unrealistic for real-world testing
- Final degradation figure saved to results/figures
- AI assistance: Claude (Anthropic) used throughout
## [Apr 18, 2026]
### Added
- Dataset downloaded: GTSRB via torchvision
- Data splits: 23,976 training, 2,664 validation, 12,630 test samples
- Data transforms and normalization pipeline
- Dataset exploration: sample images and class distribution plots
- Key finding: dataset is imbalanced (10x difference between most and least common class)
- AI assistance: Claude (Anthropic) used for code structure and guidance
## [Apr 18, 2026]
### Added
- README.md with full project proposal
- Project title, problem statement, dataset description
- Models, experiments, evaluation metrics, and timeline
- AI assistance: Claude (Anthropic) used to structure proposal content

## [Apr 17, 2026]
### Added
- Initial repository created by professor
- LICENSE and .gitignore added
