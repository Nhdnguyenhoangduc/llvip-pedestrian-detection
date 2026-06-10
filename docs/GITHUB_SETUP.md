# GitHub Setup Guide — llvip-pedestrian-detection

## 1. Repository Information

| Field | Value |
|---|---|
| **Tên repository** | `llvip-pedestrian-detection` |
| **Mô tả** | Low-light pedestrian detection via visible-infrared pixel fusion + YOLOv8s fine-tuning on the LLVIP benchmark dataset |
| **Visibility** | Public |
| **GitHub Topics** | `yolov8`, `pedestrian-detection`, `infrared-fusion`, `computer-vision`, `low-light`, `llvip`, `ultralytics`, `object-detection`, `multimodal`, `kaggle` |

---

## 2. Lệnh Git cần chạy (theo thứ tự)

```bash
# Bước 1: Khởi tạo repository
cd llvip-pedestrian-detection
git init
git branch -M main

# Bước 2: Stage tất cả file
git add README.md
git add MODEL_CARD.md
git add PROJECT_STRUCTURE.md
git add CHANGELOG.md
git add CONTRIBUTING.md
git add LICENSE
git add requirements.txt
git add .gitignore
git add configs/data.yaml
git add scripts/gradio_demo.py
git add docs/results/.gitkeep
git add assets/.gitkeep

# Bước 3: Strip notebook outputs trước khi commit
# (đã thực hiện — outputs đã được xóa)
git add notebooks/ai-predict-humman.ipynb

# Bước 4: Commit đầu tiên
git commit -m "feat: initial release v1.0.0 — LLVIP visible-infrared fusion + YOLOv8s"

# Bước 5: Kết nối remote
git remote add origin https://github.com/<your-username>/llvip-pedestrian-detection.git

# Bước 6: Push
git push -u origin main

# Bước 7: Tạo tag release
git tag -a v1.0.0 -m "Release v1.0.0 — LLVIP Pedestrian Detection"
git push origin v1.0.0
```

---

## 3. Danh sách commit đề xuất (theo lịch sử thực tế)

```
feat: initial release v1.0.0 — LLVIP visible-infrared fusion + YOLOv8s
docs: add README with full pipeline documentation
docs: add MODEL_CARD with hyperparameters and limitations
docs: add CONTRIBUTING guide and PROJECT_STRUCTURE
chore: add .gitignore for Python AI project (weights, data, cache)
feat: add standalone Gradio demo script (scripts/gradio_demo.py)
chore: add requirements.txt from notebook imports
```

---

## 4. Sau khi push — việc cần làm trên GitHub

1. **Settings → About** → Dán mô tả và topics vào
2. **Releases → Draft new release** → Tag `v1.0.0`, đính kèm `best.pt` nếu < 2GB
3. **Upload ảnh vào `docs/results/`** (xem danh sách bên dưới)
4. **Tạo Kaggle Dataset** chứa `best.pt` và link vào README

---

## 5. Release v1.0.0 — Nội dung đề xuất

**Title**: `v1.0.0 — LLVIP Low-Light Pedestrian Detection`

**Release notes**:
```
## What's included

- Full visible-infrared image fusion pipeline (alpha blending α=β=0.5)
- VOC XML → YOLO annotation converter
- YOLOv8s fine-tuning on LLVIP dataset (50 epochs, AdamW, cosine LR)
- Model evaluation: Precision, Recall, mAP@50, mAP@50-95, F1
- Gradio interactive demo (single image + paired visible/IR modes)
- Offline metrics extraction cell (no re-training required)

## Assets
- best.pt — best checkpoint by mAP@50
- last.pt — final epoch checkpoint

## How to run
See README.md for full instructions.
```
