# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [1.0.0] — 2024

### Added

- Full visible-infrared image fusion pipeline using alpha blending (α=β=0.5)
- VOC XML to YOLO format annotation converter with bounding box normalization and clipping
- YOLOv8s fine-tuning pipeline on LLVIP fused dataset
- 85/15 train/validation split strategy from LLVIP official training set
- Model evaluation pipeline using Ultralytics metrics (Precision, Recall, mAP@50, mAP@50-95, F1)
- Qualitative fusion visualization with bounding box overlays
- Inference visualization on random validation samples
- Final summary report cell with dataset and training statistics
- Gradio interactive demo supporting single image, paired image, and batch inference modes
- Offline metrics extraction cell (reads from saved `runs/` directory without re-running training)
- Environment setup with automatic package installation for Kaggle compatibility
