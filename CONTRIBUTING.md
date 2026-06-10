# Contributing Guide

Thank you for your interest in contributing to this project!

---

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/llvip-pedestrian-detection.git
   cd llvip-pedestrian-detection
   ```
3. Create a branch for your change:
   ```bash
   git checkout -b feature/your-feature-name
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## What to Contribute

- **Bug fixes** — Open an issue first describing the bug, then submit a PR.
- **New fusion methods** — e.g. Laplacian pyramid, DenseFuse, deep feature-level fusion.
- **Additional model variants** — YOLOv8n, YOLOv8m, YOLOv9, RT-DETR.
- **Multi-dataset support** — KAIST, CVC-14 adaptation.
- **Documentation improvements** — Clearer explanations, diagrams, result tables.
- **Reproducibility** — Scripts to automate dataset download and preprocessing.

---

## Notebook Guidelines

Before committing a Jupyter notebook:

```bash
# Clear all outputs to avoid committing large binary blobs
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb
```

Never commit notebooks with executed cell outputs (images, logs, model weights embedded in outputs).

---

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use type hints where practical.
- Use descriptive variable names (the existing codebase follows snake_case).
- Add docstrings to all new functions.

---

## Pull Request Process

1. Ensure your branch is up to date with `main`.
2. Clear notebook outputs before committing.
3. Update `README.md` or `CHANGELOG.md` if your change affects usage.
4. Open a PR with a clear title and description of what changed and why.
5. One reviewer approval is required to merge.

---

## Issues

Use GitHub Issues to report bugs or request features. Include:
- Python version and OS
- Exact error message and traceback
- Steps to reproduce
