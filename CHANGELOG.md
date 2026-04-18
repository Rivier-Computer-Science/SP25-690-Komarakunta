# Changelog
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
