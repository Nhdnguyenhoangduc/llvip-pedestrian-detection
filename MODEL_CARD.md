# Model Card — LLVIP Pedestrian Detector (YOLOv8s)

## Model Overview

| Field | Value |
|---|---|
| **Model Name** | `llvip_fusion_yolov8s` |
| **Base Architecture** | YOLOv8s (Ultralytics) |
| **Task** | Object Detection — Single class (person) |
| **Modality Input** | Fused visible-infrared images (3-channel BGR) |
| **Input Resolution** | 640 × 640 px |
| **Output** | Bounding boxes + confidence scores |
| **Inference Threshold** | conf = 0.25, IoU = 0.45 |

---

## Intended Use

**Intended for:**
- Nighttime / low-light pedestrian detection in research contexts
- Academic benchmarking on the LLVIP dataset
- Prototyping multimodal fusion pipelines

**Not intended for:**
- Production safety-critical systems (e.g., autonomous vehicles, surveillance) without further validation
- Detecting non-pedestrian classes
- Deployment on scenes significantly different from LLVIP's nighttime outdoor conditions

---

## Dataset

| Property | Value |
|---|---|
| **Name** | LLVIP (Low-Light Visible-Infrared Paired) |
| **Source** | [bupt-ai-cz.github.io/LLVIP](https://bupt-ai-cz.github.io/LLVIP/) |
| **Total Paired Images** | 15,488 scenes (30,976 images across modalities) |
| **Training Split** | 85% of LLVIP official train |
| **Validation Split** | 15% of LLVIP official train + LLVIP test |
| **Classes** | 1 (`person`, includes `pedestrian`) |
| **Scene Type** | Outdoor nighttime, urban environments |
| **Annotation Format** | VOC XML → converted to YOLO |

---

## Model Architecture

- **Backbone**: CSPDarknet with C2f bottleneck modules
- **Neck**: PANet (Path Aggregation Network) for multi-scale feature fusion
- **Head**: Anchor-free decoupled detection head
- **Parameters**: ~11.2M
- **Pretrained Weights**: COCO (80 classes), fine-tuned on LLVIP (1 class)

---

## Hyperparameters

| Hyperparameter | Value |
|---|---|
| Epochs | 50 |
| Batch Size | 16 |
| Image Size | 640 |
| Optimizer | AdamW |
| Initial LR | 0.01 |
| Final LR ratio | 0.001 |
| LR Schedule | Cosine Annealing |
| Weight Decay | 5e-4 |
| Warmup Epochs | 3 |
| Early Stopping Patience | 15 |

---

## Evaluation Metrics

> Fill in from your actual training run.

| Metric | `best.pt` (epoch 32) | `last.pt` (epoch 50) |
|---|---|---|
| Precision | **0.9515** | 0.9515 |
| Recall | **0.8958** | 0.8958 |
| mAP@50 | **0.9588** | 0.9588 |
| mAP@50-95 | **0.6319** | 0.6319 |
| F1-Score | **0.9228** | 0.9228 |

> Best F1 = 0.92 at confidence threshold 0.318. Precision peaks at 1.00 (conf 0.863). Recall peaks at 0.98 (conf → 0).

### Detection Performance
![Metrics](docs/results/offline_metrics_bar.png)

### Training Curves
![Training](docs/results/offline_training_curves.png)

---

## Limitations

1. **Domain specificity**: The model is trained exclusively on LLVIP nighttime outdoor scenes and may not generalize to other environments (indoor, daytime, fog, rain).
2. **Fusion simplicity**: Alpha blending (α=β=0.5) is a naive fusion approach. Scenes with high visible-IR misalignment may confuse the detector.
3. **Single class**: Only `person` is detected. Multi-class scenarios are unsupported.
4. **Small object performance**: Pedestrians smaller than ~20px in height are frequently missed.
5. **Occlusion sensitivity**: Heavily occluded pedestrians (>70% body hidden) show reduced detection reliability.

---

## Bias and Fairness

- The LLVIP dataset was collected in a specific urban environment in China. Detection performance may be biased toward the body proportions and clothing styles dominant in that dataset.
- No explicit demographic analysis was performed.
- Infrared imaging captures body heat signatures, which may differ across individuals based on clothing and ambient temperature.

---

## Recommendations for Use

- Always validate on your target domain before deployment.
- Consider adjusting the confidence threshold based on your false-positive vs. false-negative tolerance.
- For safety-critical applications, pair with a secondary validation model.
- Use `best.pt` for highest-accuracy inference; `last.pt` may be useful for analyzing final-epoch behavior.

---

## Training Environment

| Property | Value |
|---|---|
| Platform | Kaggle Notebooks |
| Accelerator | NVIDIA Tesla T4 / P100 |
| Framework | PyTorch + Ultralytics 8.x |
| CUDA | 11.x / 12.x |
