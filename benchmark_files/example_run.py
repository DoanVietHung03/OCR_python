"""
example_run.py — Ví dụ tích hợp pipeline thực tế
==================================================
Plug PARSeq + YOLO vào benchmark_alpr.py
"""

import numpy as np
import torch
import cv2
from benchmark_alpr import (
    ALPRBenchmark,
    load_test_dataset,
    load_ocr_crops,
    PipelineResult,
    print_results,
    save_results,
)

# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 1: Load model của bạn
# ─────────────────────────────────────────────────────────────────────────────

# --- Vehicle detector (YOLO) ---
# from ultralytics import YOLO
# vehicle_detector = YOLO("path/to/vehicle_detect.pt")

# --- Plate detector (YOLO) ---
# plate_detector = YOLO("path/to/plate_detect.pt")

# --- PARSeq OCR ---
# import sys
# sys.path.append("path/to/parseq")
# from strhub.data.module import SceneTextDataModule
# from strhub.models.utils import load_from_checkpoint
# parseq = load_from_checkpoint("path/to/parseq_finetuned.ckpt").eval().cuda()
# img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 2: Định nghĩa OCR function (dùng cho OCR-only benchmark)
# ─────────────────────────────────────────────────────────────────────────────

def parseq_ocr(img: np.ndarray) -> str:
    """
    Nhận vào ảnh biển (BGR numpy array),
    trả về chuỗi ký tự dự đoán.

    img đã được benchmark tự động xử lý 2 dòng → 1 dòng ngang
    nếu two_row_split=True.
    """
    # --- Thay bằng code thực tế của bạn ---
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # from PIL import Image
    # pil_img = Image.fromarray(img_rgb)
    # tensor  = img_transform(pil_img).unsqueeze(0).cuda()
    # with torch.no_grad():
    #     logits = parseq(tensor)
    #     pred   = logits.softmax(-1)
    #     label, _ = parseq.tokenizer.decode(pred)
    # return label[0]

    # Placeholder (thay bằng model thực):
    return "PLACEHOLDER"


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 3: Định nghĩa full pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def full_pipeline(img: np.ndarray) -> PipelineResult:
    """
    Nhận ảnh gốc (nhiều xe), trả về PipelineResult
    với tất cả biển số detect được.
    """
    import time
    t0 = time.perf_counter()

    pred_boxes, pred_scores, pred_texts = [], [], []
    H, W = img.shape[:2]

    # --- Thay bằng code thực tế ---

    # 1. Detect vehicle
    # vehicle_results = vehicle_detector(img)[0]
    # for vbox in vehicle_results.boxes:
    #     vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
    #     vehicle_crop = img[vy1:vy2, vx1:vx2]

    #     # 2. Detect plate trong vùng xe
    #     plate_results = plate_detector(vehicle_crop)[0]
    #     for pbox in plate_results.boxes:
    #         conf = float(pbox.conf[0])
    #         px1, py1, px2, py2 = map(int, pbox.xyxy[0])
    #         plate_crop = vehicle_crop[py1:py2, px1:px2]

    #         # 3. Preprocess + OCR
    #         text = parseq_ocr(plate_crop)

    #         # 4. Convert coords back to original image space
    #         abs_box = [vx1+px1, vy1+py1, vx1+px2, vy1+py2]
    #         pred_boxes.append(abs_box)
    #         pred_scores.append(conf)
    #         pred_texts.append(text)

    latency_ms = (time.perf_counter() - t0) * 1000

    return PipelineResult(
        pred_boxes  = pred_boxes,
        pred_scores = pred_scores,
        pred_texts  = pred_texts,
        latency_ms  = latency_ms,
    )


# ─────────────────────────────────────────────────────────────────────────────
# BƯỚC 4: Chạy benchmark
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ALPR Benchmark Runner")
    parser.add_argument("--mode",       choices=["ocr", "full"], default="ocr",
                        help="ocr = chỉ test PARSeq trên crops; full = full pipeline")
    parser.add_argument("--data",       default="test_data/",
                        help="Thư mục test set (images/ + labels/)")
    parser.add_argument("--crops",      default="test_data/crops/",
                        help="Thư mục crops (dùng cho --mode ocr)")
    parser.add_argument("--output",     default="benchmark_results.json")
    parser.add_argument("--two-row",    action="store_true", default=True,
                        help="Tự động ghép biển 2 dòng → 1 dòng trước OCR")
    args = parser.parse_args()

    bench = ALPRBenchmark(
        pipeline_fn  = full_pipeline,
        two_row_split = args.two_row,
    )

    if args.mode == "ocr":
        print("\n[Mode] OCR-only benchmark (PARSeq trên ảnh biển đã crop)")
        crops  = load_ocr_crops(args.crops)
        result = bench.run_ocr_only(parseq_ocr, crops)
        print_results(result, "OCR Benchmark — PARSeq (Vietnamese LP)")

    else:
        print("\n[Mode] Full pipeline benchmark")
        samples = load_test_dataset(args.data)
        result  = bench.run_full(samples)
        print_results(result, "Full Pipeline Benchmark")

    save_results(result, args.output)