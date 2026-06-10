"""
Standalone Gradio Demo — LLVIP Low-Light Pedestrian Detection
Extracted from the training notebook for standalone execution.

Usage:
    python scripts/gradio_demo.py --model_path path/to/best.pt
"""

import argparse
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO

# ── Defaults ─────────────────────────────────────────────────
DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
FUSION_ALPHA = 0.5
FUSION_BETA = 0.5


def load_model(model_path: str) -> YOLO:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return YOLO(str(path))


def fuse_visible_infrared(visible_bgr: np.ndarray, infrared_bgr: np.ndarray) -> np.ndarray:
    h, w = visible_bgr.shape[:2]
    if infrared_bgr.shape[:2] != (h, w):
        infrared_bgr = cv2.resize(infrared_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.addWeighted(visible_bgr, FUSION_ALPHA, infrared_bgr, FUSION_BETA, 0)


def run_inference(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float,
    iou: float,
) -> tuple[np.ndarray, int]:
    results = model.predict(
        source=image_bgr,
        conf=conf,
        iou=iou,
        imgsz=DEFAULT_IMGSZ,
        verbose=False,
    )
    annotated = results[0].plot()
    n_detections = len(results[0].boxes)
    return annotated, n_detections


def build_gradio_interface(model: YOLO) -> gr.Blocks:
    def predict_single(image_rgb: np.ndarray, conf: float, iou: float):
        if image_rgb is None:
            return None, "No image provided."
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        annotated_bgr, n = run_inference(model, image_bgr, conf, iou)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return annotated_rgb, f"Detected {n} person(s)"

    def predict_paired(visible_rgb: np.ndarray, infrared_rgb: np.ndarray, conf: float, iou: float):
        if visible_rgb is None or infrared_rgb is None:
            return None, "Please upload both visible and infrared images."
        visible_bgr = cv2.cvtColor(visible_rgb, cv2.COLOR_RGB2BGR)
        infrared_bgr = cv2.cvtColor(infrared_rgb, cv2.COLOR_RGB2BGR)
        if len(infrared_bgr.shape) == 2:
            infrared_bgr = cv2.cvtColor(infrared_bgr, cv2.COLOR_GRAY2BGR)
        fused_bgr = fuse_visible_infrared(visible_bgr, infrared_bgr)
        annotated_bgr, n = run_inference(model, fused_bgr, conf, iou)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        return annotated_rgb, f"Detected {n} person(s) in fused image"

    with gr.Blocks(title="LLVIP Pedestrian Detector") as demo:
        gr.Markdown("# 🔦 Low-Light Pedestrian Detector\nYOLOv8s fine-tuned on LLVIP visible-infrared fusion")

        with gr.Tabs():
            with gr.TabItem("Single Image"):
                with gr.Row():
                    img_in = gr.Image(label="Input Image (fused or visible)", type="numpy")
                    img_out = gr.Image(label="Detection Result", type="numpy")
                with gr.Row():
                    conf_slider = gr.Slider(0.1, 0.9, value=DEFAULT_CONF, label="Confidence Threshold")
                    iou_slider = gr.Slider(0.1, 0.9, value=DEFAULT_IOU, label="IoU Threshold")
                result_text = gr.Textbox(label="Result")
                btn = gr.Button("Detect", variant="primary")
                btn.click(predict_single, inputs=[img_in, conf_slider, iou_slider], outputs=[img_out, result_text])

            with gr.TabItem("Paired Visible + Infrared"):
                with gr.Row():
                    vis_in = gr.Image(label="Visible (RGB)", type="numpy")
                    ir_in = gr.Image(label="Infrared (Thermal)", type="numpy")
                    fused_out = gr.Image(label="Fused Detection", type="numpy")
                with gr.Row():
                    conf_slider2 = gr.Slider(0.1, 0.9, value=DEFAULT_CONF, label="Confidence Threshold")
                    iou_slider2 = gr.Slider(0.1, 0.9, value=DEFAULT_IOU, label="IoU Threshold")
                result_text2 = gr.Textbox(label="Result")
                btn2 = gr.Button("Fuse & Detect", variant="primary")
                btn2.click(
                    predict_paired,
                    inputs=[vis_in, ir_in, conf_slider2, iou_slider2],
                    outputs=[fused_out, result_text2],
                )

    return demo


def main():
    parser = argparse.ArgumentParser(description="LLVIP Pedestrian Detector — Gradio Demo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to best.pt")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path)

    demo = build_gradio_interface(model)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
