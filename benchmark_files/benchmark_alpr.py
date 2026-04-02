"""
ALPR Benchmark Script — Full Pipeline
======================================
Pipeline: raw image → vehicle detect → crop → plate detect → crop → preprocess → PARSeq OCR

Metrics:
  - Character Accuracy (CA)
  - Plate Accuracy (PA)
  - CER (Char Error Rate)
  - Detection mAP@0.5 / mAP@0.5:0.95
  - End-to-End Accuracy
  - FPS / Throughput

Cấu trúc thư mục test set cần có:
  test_data/
  ├── images/           # ảnh gốc (nhiều xe)
  ├── labels/           # YOLO format: plate bbox + text
  │   └── img001.txt    # mỗi dòng: class cx cy w h plate_text
  └── crops/            # (tuỳ chọn) ảnh biển đã crop sẵn để test OCR riêng
      ├── img001_0.jpg
      └── img001_0.txt  # nội dung: chuỗi biển số GT (vd: 51G12345)

Định dạng label (labels/*.txt):
  0 0.512 0.431 0.231 0.089 51G12345
  (class cx cy w h plate_text)
"""

import os
import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import cv2
import torch
from tqdm import tqdm

# ─── Levenshtein / edit distance ────────────────────────────────────────────
def edit_distance(s1: str, s2: str) -> int:
    """Tính Levenshtein distance giữa 2 chuỗi."""
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_cer(pred: str, gt: str) -> float:
    """CER = edit_distance / len(gt). Trả về 0.0 nếu gt rỗng."""
    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(pred, gt) / len(gt)


# ─── IoU ────────────────────────────────────────────────────────────────────
def iou(box_a: List[float], box_b: List[float]) -> float:
    """
    box format: [x1, y1, x2, y2] (pixel coords)
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ─── mAP computation ────────────────────────────────────────────────────────
def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Tính AP theo 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = precisions[recalls >= t]
        ap += np.max(p) if p.size > 0 else 0.0
    return ap / 11.0


def compute_map(
    all_preds: List[Dict],   # [{"boxes": [[x1,y1,x2,y2,...]], "scores": [...]}]
    all_gts:   List[Dict],   # [{"boxes": [[x1,y1,x2,y2],...]}]
    iou_thresholds: List[float] = None
) -> Dict[str, float]:
    """
    Tính mAP@0.5 và mAP@0.5:0.95.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5]

    aps = {}
    for iou_thresh in iou_thresholds:
        tp_list, fp_list, scores_list = [], [], []
        n_gt_total = 0

        for preds, gts in zip(all_preds, all_gts):
            gt_boxes  = gts.get("boxes", [])
            pred_boxes  = preds.get("boxes", [])
            pred_scores = preds.get("scores", [])
            n_gt_total += len(gt_boxes)

            matched = [False] * len(gt_boxes)
            # sắp xếp pred theo score giảm dần
            order = np.argsort(pred_scores)[::-1]
            for idx in order:
                pb = pred_boxes[idx]
                sc = pred_scores[idx]
                best_iou, best_j = 0.0, -1
                for j, gb in enumerate(gt_boxes):
                    v = iou(pb, gb)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= iou_thresh and not matched[best_j]:
                    tp_list.append(1); fp_list.append(0)
                    matched[best_j] = True
                else:
                    tp_list.append(0); fp_list.append(1)
                scores_list.append(sc)

        if not scores_list:
            aps[iou_thresh] = 0.0
            continue

        order = np.argsort(scores_list)[::-1]
        tp_arr = np.array(tp_list)[order]
        fp_arr = np.array(fp_list)[order]
        tp_cum = np.cumsum(tp_arr)
        fp_cum = np.cumsum(fp_arr)
        recalls    = tp_cum / (n_gt_total + 1e-8)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
        aps[iou_thresh] = compute_ap(recalls, precisions)

    map50    = aps.get(0.5, 0.0)
    map50_95 = np.mean([aps.get(t, 0.0)
                        for t in np.arange(0.5, 1.0, 0.05)])
    return {"mAP@0.5": map50, "mAP@0.5:0.95": map50_95}


# ─── Data structures ─────────────────────────────────────────────────────────
@dataclass
class Sample:
    image_path: str
    gt_boxes:   List[List[float]]   # [[x1,y1,x2,y2], ...]  pixel coords
    gt_texts:   List[str]           # ["51G12345", ...]  một GT per box


@dataclass
class PipelineResult:
    pred_boxes:  List[List[float]]
    pred_scores: List[float]
    pred_texts:  List[str]
    latency_ms:  float   # toàn bộ pipeline từ đầu vào đến kết quả cuối


# ─── Dataset loader ──────────────────────────────────────────────────────────
def load_test_dataset(data_dir: str) -> List[Sample]:
    """
    Đọc test set từ thư mục có cấu trúc:
      images/*.jpg  +  labels/*.txt
    Format label: class cx cy w h plate_text  (normalized coords)
    """
    data_dir = Path(data_dir)
    img_dir = data_dir / "images"
    lbl_dir = data_dir / "labels"

    samples = []
    for img_path in sorted(img_dir.glob("*.[jp][pn]g")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        gt_boxes, gt_texts = [], []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            # parts: class cx cy w h plate_text
            cx, cy, w, h = map(float, parts[1:5])
            text = parts[5].upper().replace("-", "").replace(".", "")
            x1 = (cx - w / 2) * W
            y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W
            y2 = (cy + h / 2) * H
            gt_boxes.append([x1, y1, x2, y2])
            gt_texts.append(text)

        if gt_boxes:
            samples.append(Sample(str(img_path), gt_boxes, gt_texts))

    print(f"[Dataset] Loaded {len(samples)} samples from {data_dir}")
    return samples


# ─── OCR-only benchmark (dùng khi đã có crops sẵn) ──────────────────────────
def load_ocr_crops(crops_dir: str) -> List[Tuple[str, str]]:
    """
    Load ảnh biển đã crop + GT text.
    Mỗi ảnh crop có file .txt cùng tên chứa chuỗi biển số GT.
    Returns: [(image_path, gt_text), ...]
    """
    crops_dir = Path(crops_dir)
    pairs = []
    for img_path in sorted(crops_dir.glob("*.[jp][pn]g")):
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        gt_text = txt_path.read_text().strip().upper().replace("-", "").replace(".", "")
        pairs.append((str(img_path), gt_text))
    print(f"[OCR Crops] Loaded {len(pairs)} cropped plates from {crops_dir}")
    return pairs


# ─── Benchmark runner ────────────────────────────────────────────────────────
class ALPRBenchmark:
    def __init__(self, pipeline_fn, two_row_split: bool = True):
        """
        pipeline_fn: callable nhận vào ảnh gốc (np.ndarray),
                     trả về PipelineResult
        two_row_split: nếu True, tự động cắt đôi biển 2 dòng trước khi OCR
        """
        self.pipeline_fn   = pipeline_fn
        self.two_row_split = two_row_split

    # ── Full-pipeline benchmark ──────────────────────────────────────────────
    def run_full(self, samples: List[Sample]) -> Dict:
        all_preds_det = []
        all_gts_det   = []
        ocr_cer_list  = []
        plate_correct = 0
        e2e_correct   = 0
        total_plates  = 0
        latencies     = []

        for sample in tqdm(samples, desc="Benchmarking"):
            img = cv2.imread(sample.image_path)
            if img is None:
                continue

            t0 = time.perf_counter()
            result: PipelineResult = self.pipeline_fn(img)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            # Detection metrics
            all_preds_det.append({
                "boxes":  result.pred_boxes,
                "scores": result.pred_scores,
            })
            all_gts_det.append({"boxes": sample.gt_boxes})

            # Match predicted plates → GT (greedy by IoU)
            matched_gt = [False] * len(sample.gt_boxes)
            for pb, pt in zip(result.pred_boxes, result.pred_texts):
                best_iou_val, best_j = 0.0, -1
                for j, gb in enumerate(sample.gt_boxes):
                    v = iou(pb, gb)
                    if v > best_iou_val:
                        best_iou_val, best_j = v, j

                if best_iou_val >= 0.5 and best_j >= 0 and not matched_gt[best_j]:
                    matched_gt[best_j] = True
                    gt_text   = sample.gt_texts[best_j]
                    pred_text = pt.upper().replace("-", "").replace(".", "")
                    cer_val   = compute_cer(pred_text, gt_text)
                    ocr_cer_list.append(cer_val)
                    plate_correct += int(pred_text == gt_text)
                    # End-to-End: phải detect đúng VÀ đọc đúng
                    e2e_correct   += int(best_iou_val >= 0.5 and pred_text == gt_text)
                    total_plates  += 1

            # Các GT plate không được match coi như sai hoàn toàn
            for j, matched in enumerate(matched_gt):
                if not matched:
                    ocr_cer_list.append(1.0)
                    total_plates += 1

        map_scores = compute_map(
            all_preds_det, all_gts_det,
            iou_thresholds=list(np.arange(0.5, 1.0, 0.05))
        )
        avg_cer  = float(np.mean(ocr_cer_list)) if ocr_cer_list else 1.0
        avg_lat  = float(np.mean(latencies))
        fps      = 1000.0 / avg_lat if avg_lat > 0 else 0.0
        throughput = fps  # plates/sec ≈ FPS nếu trung bình 1 biển/ảnh

        return {
            "Detection mAP@0.5":      round(map_scores["mAP@0.5"] * 100, 2),
            "Detection mAP@0.5:0.95": round(map_scores["mAP@0.5:0.95"] * 100, 2),
            "CER (%)":                round(avg_cer * 100, 2),
            "CA — Char Accuracy (%)": round((1 - avg_cer) * 100, 2),
            "PA — Plate Accuracy (%)": round(plate_correct / total_plates * 100, 2)
                                        if total_plates > 0 else 0.0,
            "End-to-End Accuracy (%)": round(e2e_correct / total_plates * 100, 2)
                                        if total_plates > 0 else 0.0,
            "Avg Latency (ms)":       round(avg_lat, 1),
            "FPS":                    round(fps, 1),
            "Throughput (plates/s)":  round(throughput, 1),
            "Total samples":          len(samples),
            "Total plates":           total_plates,
        }

    # ── OCR-only benchmark (input = cropped plates) ──────────────────────────
    def run_ocr_only(self, ocr_fn, crops: List[Tuple[str, str]]) -> Dict:
        """
        ocr_fn: callable nhận ảnh crop (np.ndarray) → str (predicted text)
        crops:  [(image_path, gt_text), ...]
        """
        cer_list       = []
        plate_correct  = 0
        latencies      = []

        for img_path, gt_text in tqdm(crops, desc="OCR Benchmark"):
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Cắt đôi biển 2 dòng nếu cần
            if self.two_row_split:
                img = self._split_two_row(img)

            t0 = time.perf_counter()
            pred_text = ocr_fn(img).upper().replace("-", "").replace(".", "")
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            cer_val = compute_cer(pred_text, gt_text)
            cer_list.append(cer_val)
            plate_correct += int(pred_text == gt_text)

        n = len(crops)
        avg_cer = float(np.mean(cer_list)) if cer_list else 1.0
        avg_lat = float(np.mean(latencies)) if latencies else 0.0

        return {
            "CER (%)":                round(avg_cer * 100, 2),
            "CA — Char Accuracy (%)": round((1 - avg_cer) * 100, 2),
            "PA — Plate Accuracy (%)": round(plate_correct / n * 100, 2) if n > 0 else 0.0,
            "Avg OCR Latency (ms)":   round(avg_lat, 1),
            "OCR Throughput (plates/s)": round(1000 / avg_lat, 1) if avg_lat > 0 else 0.0,
            "Total crops evaluated":  n,
        }

    @staticmethod
    def _split_two_row(img: np.ndarray) -> np.ndarray:
        """
        Ghép 2 dòng của biển xe máy thành 1 dòng ngang
        để PARSeq đọc liên tục.
        """
        h, w = img.shape[:2]
        top    = img[:h // 2, :]
        bottom = img[h // 2:, :]
        # Resize bottom về cùng chiều cao với top
        bottom = cv2.resize(bottom, (top.shape[1], top.shape[0]))
        return np.concatenate([top, bottom], axis=1)


# ─── Kết quả in đẹp ──────────────────────────────────────────────────────────
def print_results(results: Dict, title: str = "Benchmark Results"):
    width = 50
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)
    for k, v in results.items():
        print(f"  {k:<35} {v}")
    print("═" * width + "\n")


def save_results(results: Dict, output_path: str = "benchmark_results.json"):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Saved] Results → {output_path}")


# ─── Entry point / hướng dẫn tích hợp ───────────────────────────────────────
"""
=== CÁCH SỬ DỤNG ===

1. OCR-ONLY (nếu bạn chỉ test PARSeq trên ảnh biển đã crop):

    from benchmark_alpr import ALPRBenchmark, load_ocr_crops, print_results

    def my_parseq_ocr(img: np.ndarray) -> str:
        # gọi model PARSeq của bạn ở đây
        # img là ảnh biển (đã xử lý 2 dòng → 1 dòng nếu two_row_split=True)
        return "51G12345"

    crops  = load_ocr_crops("test_data/crops/")
    bench  = ALPRBenchmark(pipeline_fn=None, two_row_split=True)
    result = bench.run_ocr_only(my_parseq_ocr, crops)
    print_results(result, "OCR Benchmark — PARSeq")
    save_results(result)


2. FULL PIPELINE:

    from benchmark_alpr import ALPRBenchmark, load_test_dataset, print_results, PipelineResult

    def my_full_pipeline(img: np.ndarray) -> PipelineResult:
        # 1. vehicle detect
        # 2. crop vehicle
        # 3. plate detect
        # 4. crop plate
        # 5. preprocess
        # 6. parseq OCR
        return PipelineResult(
            pred_boxes  = [[x1, y1, x2, y2], ...],
            pred_scores = [0.95, ...],
            pred_texts  = ["51G12345", ...],
            latency_ms  = 42.0,
        )

    samples = load_test_dataset("test_data/")
    bench   = ALPRBenchmark(pipeline_fn=my_full_pipeline, two_row_split=False)
    result  = bench.run_full(samples)
    print_results(result, "Full Pipeline Benchmark")
    save_results(result)
"""

if __name__ == "__main__":
    print(__doc__)
    print("Vui lòng import module này và tích hợp pipeline của bạn.")
    print("Xem phần '=== CÁCH SỬ DỤNG ===' ở cuối file.")