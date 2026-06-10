# 🔦 Low-Light Pedestrian Detection via Visible-Infrared Fusion + YOLOv8

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset: LLVIP](https://img.shields.io/badge/Dataset-LLVIP-red.svg)](https://bupt-ai-cz.github.io/LLVIP/)
[![Platform: Kaggle](https://img.shields.io/badge/Platform-Kaggle-20BEFF.svg)](https://www.kaggle.com/)

Pedestrian detection in low-light environments using **pixel-level image fusion** of visible (RGB) and infrared (IR) modalities, fine-tuning **YOLOv8s** on the **LLVIP** benchmark dataset.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training Pipeline](#training-pipeline)
- [Evaluation Metrics](#evaluation-metrics)
- [Experimental Results](#experimental-results)
- [Installation](#installation)
- [Usage](#usage)
  - [Run Training](#run-training)
  - [Run Inference](#run-inference)
  - [Retrain from Scratch](#retrain-from-scratch)
- [Project Structure](#project-structure)
- [Gradio Demo](#gradio-demo)
- [Limitations](#limitations)
- [Future Work](#future-work)

---

## Introduction

Standard RGB cameras degrade significantly in darkness, making nighttime pedestrian detection a critical challenge for autonomous driving, surveillance, and public safety systems. Infrared cameras, however, capture thermal radiation and remain effective regardless of illumination.

This project proposes a **dual-modality fusion pipeline** that blends visible-light and infrared images at the pixel level (weighted alpha blending), then trains **YOLOv8s** — a state-of-the-art real-time object detector — on the fused images using the **LLVIP** (Low-Light Visible-Infrared Paired) dataset.

---

## Objectives

- Implement a reproducible multimodal image fusion pipeline for low-light pedestrian detection.
- Convert VOC-format annotations to YOLO format for seamless integration with Ultralytics.
- Fine-tune YOLOv8s on fused visible-infrared images.
- Evaluate detection performance using standard COCO metrics (mAP@50, mAP@50-95, Precision, Recall, F1).
- Provide a Gradio-based interactive demo for real-time inference.

---

## System Architecture

```
LLVIP Dataset
├── Visible (RGB) images     ─────────────────────────────────────────┐
└── Infrared (IR) images     ────── Alpha Blend (α=0.5, β=0.5) ──────► Fused Image
                                                                       │
LLVIP Annotations (VOC XML) ──── parse_voc_xml_to_yolo() ────────────► YOLO Labels (.txt)
                                                                       │
                              YOLOv8s Pretrained (COCO)                │
                                     │                                 │
                                     └─── Fine-tune on Fused Data ◄───┘
                                                   │
                                          best.pt / last.pt
                                                   │
                              Inference (conf=0.25, IoU=0.45)
                                                   │
                                       Detection Results + Gradio Demo
```

---

## Dataset

**LLVIP** (Low-Light Visible-Infrared Paired) is a benchmark dataset specifically designed for person detection in low-light conditions.

| Property | Value |
|---|---|
| Total Images | 30,976 paired images (15,488 scenes) |
| Training Set | ~12,025 images (85% of LLVIP train split) |
| Validation Set | ~2,122 images (15% of LLVIP train split + official test) |
| Resolution | 1280 × 1024 pixels |
| Modalities | Visible (RGB) + Infrared (thermal) |
| Classes | 1 — `person` (includes `pedestrian`) |
| Annotation Format | Pascal VOC XML |
| Source | [LLVIP Official](https://bupt-ai-cz.github.io/LLVIP/) |

All paired images share identical filenames across visible and infrared directories.

---

## Data Preprocessing

### 1. Image Fusion

Each visible-infrared pair is fused using **weighted alpha blending**:

```
Fused = α × Visible_BGR + β × Infrared_as_BGR
```

- `α = 0.5`, `β = 0.5` (equal contribution from both modalities)
- Infrared images (grayscale) are converted to 3-channel BGR before blending
- Images are spatially aligned by resizing infrared to match visible resolution

### 2. Annotation Conversion

VOC XML annotations are converted to YOLO format:

```
<class_id> <cx> <cy> <bw> <bh>   (all values normalized to [0,1])
```

- Only `person` and `pedestrian` class labels are retained (mapped to class ID `0`)
- Bounding boxes are clipped to image boundaries before normalization

### 3. Train/Val Split

- 85% of the LLVIP official training set → training
- 15% of the LLVIP official training set → validation
- Official LLVIP test set used for final evaluation

---

## Model Architecture

**YOLOv8s** (Small variant of YOLOv8 by Ultralytics):

| Component | Details |
|---|---|
| Backbone | CSPDarknet with C2f modules |
| Neck | PANet (Path Aggregation Network) |
| Head | Decoupled detection head |
| Input Size | 640 × 640 |
| Parameters | ~11.2M |
| Pretrained Weights | COCO (80 classes) |
| Fine-tuned Classes | 1 (`person`) |

---

## Training Pipeline

| Hyperparameter | Value |
|---|---|
| Epochs | 50 |
| Batch Size | 16 |
| Image Size | 640 |
| Optimizer | AdamW |
| Initial LR (`lr0`) | 0.01 |
| Final LR (`lrf`) | 0.001 |
| LR Schedule | Cosine annealing |
| Weight Decay | 5e-4 |
| Warmup Epochs | 3 |
| Early Stopping Patience | 15 |
| Workers | 4 |
| Pretrained | Yes (COCO) |

---

## Evaluation Metrics

Metrics are computed using the Ultralytics validation pipeline (COCO-standard):

| Metric | Description |
|---|---|
| **Precision** | TP / (TP + FP) at IoU ≥ 0.50 |
| **Recall** | TP / (TP + FN) at IoU ≥ 0.50 |
| **mAP@50** | Mean Average Precision at IoU = 0.50 |
| **mAP@50-95** | Mean Average Precision across IoU thresholds [0.50:0.05:0.95] |
| **F1-Score** | Harmonic mean of Precision and Recall |

Inference thresholds: `conf=0.25`, `iou=0.45`.

---

## Experimental Results

> YOLOv8s fine-tuned on the LLVIP visible-infrared fused dataset — **best checkpoint at epoch 32** (50 epochs, AdamW, cosine LR).

| Metric | Value |
|---|---|
| **Precision** | **0.9515** |
| **Recall** | **0.8958** |
| **mAP@50** | **0.9588** |
| **mAP@50-95** | **0.6319** |
| **F1-Score** | **0.9228** |

> Best confidence threshold for F1: **0.318** — Precision reaches 1.00 at conf 0.863 — Recall reaches 0.98 at low confidence.

### Detection Performance Summary
![Metrics Bar Chart](docs/results/offline_metrics_bar.png)

### Training & Validation Curves
![Training Curves](docs/results/offline_training_curves.png)

### Evaluation Curves (PR / F1 / Precision / Recall)
![Ultralytics Evaluation Plots](docs/results/offline_ultralytics_plots.png)

### Confusion Matrix
| Raw counts | Normalized |
|---|---|
| ![Confusion Matrix](docs/results/confusion_matrix.png) | ![Confusion Matrix Normalized](docs/results/confusion_matrix_normalized.png) |

### Fusion Visualization (Visible vs Infrared vs Fused + GT Boxes)
![Fusion Samples](assets/fusion_samples_visualization.png)

---

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended, tested on Kaggle T4/P100)
- 16GB+ RAM

### Setup

```bash
git clone https://github.com/<your-username>/llvip-pedestrian-detection.git
cd llvip-pedestrian-detection

pip install -r requirements.txt
```

---

## Usage

### Dataset Preparation

Download the LLVIP dataset from the [official source](https://bupt-ai-cz.github.io/LLVIP/) and organize as follows:

```
data/
└── LLVIP/
    ├── visible/
    │   ├── train/   # RGB training images
    │   └── test/    # RGB test images
    ├── infrared/
    │   ├── train/   # IR training images
    │   └── test/    # IR test images
    └── Annotations/ # VOC XML files
```

Update paths in the notebook or `configs/paths.yaml`.

### Run Training

Open and run the notebook end-to-end:

```bash
jupyter notebook notebooks/ai-predict-humman.ipynb
```

Or run on Kaggle by uploading the notebook and attaching the LLVIP dataset.

### Run Inference

```python
from ultralytics import YOLO
import cv2

model = YOLO("path/to/best.pt")

# Single image inference
results = model.predict(
    source="path/to/fused_image.jpg",
    conf=0.25,
    iou=0.45,
    imgsz=640,
)
annotated = results[0].plot()
cv2.imwrite("output.jpg", annotated)
```

### Retrain from Scratch

1. Run Cells 1–4 to set up environment, fuse images, and create YOLO labels.
2. Run Cell 6 to start fine-tuning YOLOv8s.
3. Run Cell 7 for evaluation.
4. Run Cell 8 to visualize inference samples.

### Launch Gradio Demo

```bash
# After training, run the Gradio cell (Cell GUI) in the notebook
# Or run standalone:
python scripts/gradio_demo.py --model_path runs/llvip_fusion_yolov8s/weights/best.pt
```

---

## Project Structure

```
llvip-pedestrian-detection/
├── notebooks/
│   └── ai-predict-humman.ipynb     # Main training & evaluation notebook
├── scripts/
│   └── gradio_demo.py              # Standalone Gradio demo script
├── configs/
│   └── data.yaml                   # YOLO dataset config (auto-generated)
├── docs/
│   ├── results/                    # Training curves, confusion matrix (add manually)
│   └── architecture.png            # System architecture diagram (add manually)
├── assets/
│   └── demo.gif                    # Demo GIF (add manually)
├── README.md
├── MODEL_CARD.md
├── PROJECT_STRUCTURE.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

## Gradio Demo

The notebook includes a full Gradio interactive web interface (Cell GUI) supporting:

- **Single image upload**: Upload a visible or fused image for detection
- **Paired upload**: Upload both visible + infrared images for on-the-fly fusion and detection
- **Configurable thresholds**: Adjust confidence and IoU sliders
- **Batch inference**: Sample random validation images and display annotated results

---

## Limitations

- The fusion strategy (alpha blending) is a simple pixel-level approach; more sophisticated methods (Laplacian pyramid, deep fusion) may yield better results.
- The model is trained exclusively on the LLVIP dataset and may not generalize to other low-light scenes without retraining.
- Performance degrades on occluded or very small pedestrians (< 20px height).
- No multi-class detection — the model detects only `person`.
- Inference speed depends on GPU availability; CPU inference is significantly slower.

---

## Future Work

- [ ] Experiment with deep learning-based fusion methods (e.g., DenseFuse, RFN-Nest)
- [ ] Extend to multi-class detection (vehicles, cyclists)
- [ ] Benchmark YOLOv8n, YOLOv8m, and YOLOv8l variants
- [ ] Add TensorRT/ONNX export for edge deployment
- [ ] Implement tracking (ByteTrack / BotSort) for video pedestrian counting
- [ ] Evaluate on additional low-light datasets (KAIST, CVC-14)
- [ ] Publish pretrained weights to Hugging Face Hub

---

## Citation

If you use this work, please cite the LLVIP dataset:

```bibtex
@InProceedings{jia2021llvip,
  title     = {LLVIP: A Visible-infrared Paired Dataset for Low-light Vision},
  author    = {Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
  year      = {2021}
}
```

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
