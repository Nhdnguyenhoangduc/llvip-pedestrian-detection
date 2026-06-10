#!/usr/bin/env python3
"""
push_to_github.py
─────────────────
Script tự động hoàn chỉnh để:
  1. Nhận metrics từ bạn (hoặc đọc từ results.csv nếu có)
  2. Điền metrics vào README.md và MODEL_CARD.md
  3. Copy ảnh vào đúng thư mục
  4. Tạo GitHub repository
  5. Git init + commit + push
  6. Tạo Release v1.0.0
  7. Gắn Topics

Chạy: python push_to_github.py
"""

import os
import sys
import json
import shutil
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

# ════════════════════════════════════════════════════════════════
# CẤU HÌNH — chỉ chỉnh sửa phần này
# ════════════════════════════════════════════════════════════════

GITHUB_TOKEN    = ""
GITHUB_USERNAME = "Nhdnguyenhoangduc"
REPO_NAME       = "llvip-pedestrian-detection"
REPO_DESC       = "Low-light pedestrian detection via visible-infrared pixel fusion + YOLOv8s fine-tuning on the LLVIP benchmark dataset"
REPO_TOPICS     = ["yolov8", "pedestrian-detection", "infrared-fusion", "low-light",
                   "llvip", "ultralytics", "object-detection", "multimodal", "kaggle", "computer-vision"]

# Thư mục chứa repo (thư mục chứa file này)
REPO_DIR = Path(__file__).parent.resolve()

# Thư mục ảnh bạn tải về từ Kaggle
# Đặt tất cả ảnh vào cùng 1 thư mục, script sẽ tự copy
IMAGES_SOURCE_DIR = REPO_DIR / "kaggle_exports"   # tạo thư mục này, bỏ ảnh vào đây

# ════════════════════════════════════════════════════════════════

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"{GREEN}✓{RESET} {msg}")
def warn(msg): print(f"{YELLOW}⚠{RESET}  {msg}")
def err(msg):  print(f"{RED}✗{RESET} {msg}")
def step(msg): print(f"\n{BOLD}{'─'*55}{RESET}\n{BOLD}{msg}{RESET}")


# ════════════════════════════════════════════════════════════════
# GitHub API helper
# ════════════════════════════════════════════════════════════════

def gh_request(method: str, endpoint: str, data: dict = None) -> dict:
    url = f"https://api.github.com{endpoint}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url, data=body, method=method,
        headers={
            "Authorization": f"token {GITHUB_TOKEN}",
            "Accept": "application/vnd.github.v3+json",
            "Content-Type": "application/json",
            "User-Agent": "llvip-push-script/1.0",
        }
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode()
        return {"error": e.code, "message": body_text}


# ════════════════════════════════════════════════════════════════
# BƯỚC 0 — Kiểm tra token
# ════════════════════════════════════════════════════════════════

def check_token() -> str:
    step("BƯỚC 0 — Kiểm tra GitHub token")
    result = gh_request("GET", "/user")
    if "error" in result:
        err(f"Token không hợp lệ hoặc hết hạn: {result}")
        sys.exit(1)
    login = result.get("login", "")
    ok(f"Token hợp lệ — đăng nhập là: {login}")
    return login


# ════════════════════════════════════════════════════════════════
# BƯỚC 1 — Đọc metrics (từ results.csv hoặc nhập tay)
# ════════════════════════════════════════════════════════════════

def read_metrics_from_csv() -> dict | None:
    csv_candidates = list(REPO_DIR.rglob("results.csv"))
    if not csv_candidates:
        return None
    csv_path = csv_candidates[0]
    ok(f"Tìm thấy results.csv: {csv_path}")
    import csv
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", "0").strip() or 0))
    
    def get(key):
        val = best.get(key, "").strip()
        try:
            return f"{float(val):.4f}"
        except:
            return "—"

    return {
        "precision": get("metrics/precision(B)"),
        "recall":    get("metrics/recall(B)"),
        "map50":     get("metrics/mAP50(B)"),
        "map5095":   get("metrics/mAP50-95(B)"),
    }


def ask_metrics() -> dict:
    step("BƯỚC 1 — Nhập kết quả training")
    print("Dán kết quả từ Kaggle (để trống nếu chưa có, nhấn Enter để bỏ qua):")

    def ask(label):
        val = input(f"  {label}: ").strip()
        return val if val else "—"

    precision = ask("Precision")
    recall    = ask("Recall")
    map50     = ask("mAP@50")
    map5095   = ask("mAP@50-95")

    f1 = "—"
    try:
        p, r = float(precision), float(recall)
        f1 = f"{2*p*r/(p+r):.4f}"
    except:
        f1 = ask("F1-Score")

    return {"precision": precision, "recall": recall,
            "map50": map50, "map5095": map5095, "f1": f1}


def fill_metrics(metrics: dict):
    step("BƯỚC 1b — Điền metrics vào README và MODEL_CARD")

    f1 = metrics.get("f1", "—")
    if f1 == "—":
        try:
            p, r = float(metrics["precision"]), float(metrics["recall"])
            f1 = f"{2*p*r/(p+r):.4f}"
        except:
            pass

    replacements = {
        "PRECISION_PLACEHOLDER": metrics["precision"],
        "RECALL_PLACEHOLDER":    metrics["recall"],
        "MAP50_PLACEHOLDER":     metrics["map50"],
        "MAP5095_PLACEHOLDER":   metrics["map5095"],
        "F1_PLACEHOLDER":        f1,
    }

    for fname in ["README.md", "MODEL_CARD.md"]:
        fpath = REPO_DIR / fname
        if not fpath.exists():
            continue
        text = fpath.read_text(encoding="utf-8")
        for placeholder, value in replacements.items():
            text = text.replace(f"**{placeholder}**", value)
            text = text.replace(placeholder, value)
            # Also replace "—" in markdown tables for MODEL_CARD
            if value != "—":
                text = text.replace("| — |", f"| {value} |", 1)
        fpath.write_text(text, encoding="utf-8")
        ok(f"Metrics điền vào {fname}")


# ════════════════════════════════════════════════════════════════
# BƯỚC 2 — Copy ảnh từ kaggle_exports vào đúng vị trí
# ════════════════════════════════════════════════════════════════

IMAGE_MAP = {
    "confusion_matrix.png":            REPO_DIR / "docs" / "results" / "confusion_matrix.png",
    "confusion_matrix_normalized.png": REPO_DIR / "docs" / "results" / "confusion_matrix_normalized.png",
    "BoxPR_curve.png":                 REPO_DIR / "docs" / "results" / "BoxPR_curve.png",
    "BoxF1_curve.png":                 REPO_DIR / "docs" / "results" / "BoxF1_curve.png",
    "BoxP_curve.png":                  REPO_DIR / "docs" / "results" / "BoxP_curve.png",
    "BoxR_curve.png":                  REPO_DIR / "docs" / "results" / "BoxR_curve.png",
    "offline_training_curves.png":     REPO_DIR / "docs" / "results" / "offline_training_curves.png",
    "offline_metrics_bar.png":         REPO_DIR / "docs" / "results" / "offline_metrics_bar.png",
    "offline_ultralytics_plots.png":   REPO_DIR / "docs" / "results" / "offline_ultralytics_plots.png",
    "fusion_samples_visualization.png": REPO_DIR / "assets" / "fusion_samples_visualization.png",
}


def copy_images():
    step("BƯỚC 2 — Copy ảnh kết quả")
    src_dir = IMAGES_SOURCE_DIR

    if not src_dir.exists():
        warn(f"Thư mục {src_dir} không tồn tại.")
        warn("Tạo thư mục 'kaggle_exports/' bên cạnh script này")
        warn("rồi bỏ các ảnh từ Kaggle vào đó, sau đó chạy lại script.")
        warn("Script vẫn tiếp tục — ảnh sẽ bị thiếu trong repo.")
        return

    copied, missing = [], []
    for filename, dest in IMAGE_MAP.items():
        src = src_dir / filename
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            copied.append(filename)
            ok(f"Copy: {filename}")
        else:
            missing.append(filename)
            warn(f"Không tìm thấy: {filename}")

    # Also copy results.csv if present
    csv_src = src_dir / "results.csv"
    if csv_src.exists():
        shutil.copy2(csv_src, REPO_DIR / "docs" / "results" / "results.csv")
        ok("Copy: results.csv")

    if missing:
        warn(f"{len(missing)} ảnh còn thiếu — có thể bổ sung sau bằng git add + git push")


# ════════════════════════════════════════════════════════════════
# BƯỚC 3 — Tạo GitHub repo
# ════════════════════════════════════════════════════════════════

def create_github_repo():
    step("BƯỚC 3 — Tạo GitHub repository")

    # Kiểm tra repo đã tồn tại chưa
    existing = gh_request("GET", f"/repos/{GITHUB_USERNAME}/{REPO_NAME}")
    if "id" in existing:
        warn(f"Repo đã tồn tại: {existing['html_url']}")
        return existing["html_url"]

    result = gh_request("POST", "/user/repos", {
        "name":        REPO_NAME,
        "description": REPO_DESC,
        "private":     False,
        "auto_init":   False,
    })

    if "error" in result:
        err(f"Tạo repo thất bại: {result}")
        sys.exit(1)

    url = result["html_url"]
    ok(f"Repo tạo thành công: {url}")
    return url


# ════════════════════════════════════════════════════════════════
# BƯỚC 4 — Git init + commit + push
# ════════════════════════════════════════════════════════════════

def run_git(args: list, cwd: Path, check=True):
    result = subprocess.run(
        ["git"] + args, cwd=str(cwd),
        capture_output=True, text=True
    )
    if check and result.returncode != 0:
        err(f"git {' '.join(args)} thất bại:\n{result.stderr}")
        sys.exit(1)
    return result


def git_push():
    step("BƯỚC 4 — Git init + commit + push")

    remote_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"

    # Remove .gitkeep files (not needed once real content is added)
    for gk in REPO_DIR.rglob(".gitkeep"):
        gk.unlink()

    run_git(["init"], REPO_DIR)
    run_git(["config", "user.email", "nhdnguyenhoangduc@github.com"], REPO_DIR)
    run_git(["config", "user.name",  GITHUB_USERNAME], REPO_DIR)
    run_git(["branch", "-M", "main"], REPO_DIR)

    # Check if remote exists
    remotes = run_git(["remote"], REPO_DIR, check=False).stdout.strip()
    if "origin" in remotes:
        run_git(["remote", "set-url", "origin", remote_url], REPO_DIR)
    else:
        run_git(["remote", "add", "origin", remote_url], REPO_DIR)

    run_git(["add", "."], REPO_DIR)

    # Check if there's anything to commit
    status = run_git(["status", "--porcelain"], REPO_DIR, check=False).stdout.strip()
    if not status:
        warn("Không có thay đổi mới để commit (repo đã được push trước đó?)")
    else:
        run_git(["commit", "-m",
                 "feat: initial release v1.0.0 — LLVIP visible-infrared fusion + YOLOv8s"],
                REPO_DIR)
        ok("Commit thành công")

    run_git(["push", "-u", "origin", "main", "--force"], REPO_DIR)
    ok(f"Push thành công → https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")


# ════════════════════════════════════════════════════════════════
# BƯỚC 5 — Tạo Release v1.0.0
# ════════════════════════════════════════════════════════════════

def create_release():
    step("BƯỚC 5 — Tạo GitHub Release v1.0.0")

    result = gh_request("POST", f"/repos/{GITHUB_USERNAME}/{REPO_NAME}/releases", {
        "tag_name":         "v1.0.0",
        "target_commitish": "main",
        "name":             "v1.0.0 — LLVIP Low-Light Pedestrian Detection",
        "body": (
            "## What's included\n\n"
            "- ✅ Full visible-infrared image fusion pipeline (alpha blending α=β=0.5)\n"
            "- ✅ VOC XML → YOLO annotation converter\n"
            "- ✅ YOLOv8s fine-tuning on LLVIP dataset (50 epochs, AdamW, cosine LR)\n"
            "- ✅ Model evaluation: Precision, Recall, mAP@50, mAP@50-95, F1\n"
            "- ✅ Gradio interactive demo (single image + paired visible/IR modes)\n"
            "- ✅ Offline metrics extraction (no re-training required)\n\n"
            "## Pretrained Weights\n\n"
            "Upload `best.pt` lên Kaggle Dataset và thêm link vào README.\n\n"
            "## How to run\n\n"
            "Xem [README.md](README.md) để biết hướng dẫn đầy đủ."
        ),
        "draft":      False,
        "prerelease": False,
    })

    if "error" in result:
        warn(f"Tạo release thất bại (có thể tag đã tồn tại): {result.get('message', '')}")
    else:
        ok(f"Release tạo thành công: {result.get('html_url', '')}")


# ════════════════════════════════════════════════════════════════
# BƯỚC 6 — Gắn Topics
# ════════════════════════════════════════════════════════════════

def set_topics():
    step("BƯỚC 6 — Gắn GitHub Topics")

    result = gh_request("PUT",
        f"/repos/{GITHUB_USERNAME}/{REPO_NAME}/topics",
        {"names": REPO_TOPICS}
    )

    if "error" in result:
        warn(f"Gắn topics thất bại: {result}")
    else:
        ok(f"Topics: {', '.join(REPO_TOPICS)}")


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{BOLD}  LLVIP Pedestrian Detection — Auto GitHub Push{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")
    print(f"  Repo : {GITHUB_USERNAME}/{REPO_NAME}")
    print(f"  Dir  : {REPO_DIR}")
    print(f"{BOLD}{'═'*55}{RESET}\n")

    # Bước 0: Kiểm tra token
    check_token()

    # Bước 1: Metrics
    metrics = read_metrics_from_csv()
    if metrics:
        ok(f"Đọc metrics từ results.csv: mAP@50 = {metrics['map50']}")
        try:
            p, r = float(metrics["precision"]), float(metrics["recall"])
            metrics["f1"] = f"{2*p*r/(p+r):.4f}"
        except:
            metrics["f1"] = "—"
    else:
        warn("Không tìm thấy results.csv — bạn có thể nhập tay hoặc bỏ qua.")
        answer = input("  Nhập metrics tay? (y/n): ").strip().lower()
        if answer == "y":
            metrics = ask_metrics()
        else:
            metrics = {"precision": "—", "recall": "—",
                       "map50": "—", "map5095": "—", "f1": "—"}
            warn("Metrics để trống — điền sau bằng cách chỉnh README.md tay")

    fill_metrics(metrics)

    # Bước 2: Copy ảnh
    copy_images()

    # Bước 3: Tạo repo
    create_github_repo()

    # Bước 4: Git push
    git_push()

    # Bước 5: Release
    create_release()

    # Bước 6: Topics
    set_topics()

    # Tổng kết
    print(f"\n{BOLD}{'═'*55}{RESET}")
    print(f"{GREEN}{BOLD}  ✅ HOÀN THÀNH!{RESET}")
    print(f"{BOLD}{'═'*55}{RESET}")
    print(f"  🔗 Repository : https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
    print(f"  📦 Release    : https://github.com/{GITHUB_USERNAME}/{REPO_NAME}/releases/tag/v1.0.0")
    print(f"\n  Việc còn lại (làm thủ công):")
    print(f"  • Upload best.pt + last.pt lên Kaggle Dataset")
    print(f"  • Thêm link weights vào README.md rồi git push lại")
    print(f"  • Thêm link Kaggle notebook vào badge trong README")
    print(f"{BOLD}{'═'*55}{RESET}\n")


if __name__ == "__main__":
    main()
