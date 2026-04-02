import cv2
import numpy as np
import supervision as sv
import os
import csv
import time
from datetime import datetime
import re
import threading  
import queue 

from config import OCRResult, WEIGHT_PLATE, WEIGHT_OCR
from utils import (
    enhance_plate_quality,
    calculate_blur_score,
    preprocess_and_normalize_ocr,
    clean_plate_text,
    clean_top_line,
    clean_bottom_line,
    apply_ir_handling,
)
from inference import infer_yolo, decode_parseq

LOG_DIR = "event_logs"
IMG_DIR = os.path.join(LOG_DIR, "images")
CSV_FILE = os.path.join(LOG_DIR, "history.csv")

os.makedirs(IMG_DIR, exist_ok=True)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Thời gian", "ID_Xe", "Biển_số", "Độ_tự_tin", "Ảnh_Xe", "Ảnh_Biển"]
        )

VEHICLE_DETECT_INTERVAL = 1
OCR_INTERVAL = 5
STALE_GRACE_FRAMES = 15

# ─── THÊM HÀNG ĐỢI VÀ LUỒNG I/O CHẠY NGẦM (ASYNC LOGGING) ────────────────────
log_queue = queue.Queue()


def async_logger_worker():
    while True:
        item = log_queue.get()
        if item is None:
            break
        try:
            veh_crop, pla_crop, t_str, track_id, text, conf_str, veh_name, pla_name = (
                item
            )

            # Ghi ảnh xe xuống ổ cứng
            cv2.imwrite(os.path.join(IMG_DIR, veh_name), veh_crop)

            # Ghi ảnh biển số xuống ổ cứng (nếu có)
            if pla_crop is not None and pla_crop.size > 0:
                cv2.imwrite(os.path.join(IMG_DIR, pla_name), pla_crop)

            # Ghi vào CSV
            with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [t_str, track_id, text, conf_str, veh_name, pla_name]
                )
        except Exception as e:
            print(f"[I/O Logger Error] {e}")
        finally:
            log_queue.task_done()


# Khởi chạy luồng I/O ngay khi import (Daemon=True để luồng tự tắt khi app tắt)
threading.Thread(target=async_logger_worker, daemon=True).start()
# ─────────────────────────────────────────────────────────────────────────────


def process_single_frame(
    img_bgr,
    vehicle_session,
    plate_session,
    parseq_session,
    vehicle_in,
    vehicle_out,
    plate_in,
    plate_out,
    parseq_in,
    parseq_out,
    allowed_vehicle_ids,
    tracker,
    ocr_cache,
    tracker_state,
    frame_id=0,
):

    # ── BƯỚC 1: Vehicle Detection ────────────────────────────────────────────
    small_img_bgr = cv2.resize(img_bgr, (0, 0), fx=0.5, fy=0.5)
    if frame_id % VEHICLE_DETECT_INTERVAL == 0:
        small_vehicle_dets = infer_yolo(
            vehicle_session,
            vehicle_in,
            vehicle_out,
            small_img_bgr,
            0.4,
            allowed_vehicle_ids,
        )
        tracker_state["last_dets"] = small_vehicle_dets
    else:
        small_vehicle_dets = tracker_state.get("last_dets", [])

    vehicle_dets = []
    for det in small_vehicle_dets:
        x, y, w, h = det.box
        det.box = [int(x * 2.0), int(y * 2.0), int(w * 2.0), int(h * 2.0)]
        vehicle_dets.append(det)

    # ── BƯỚC 2: Tracker ──────────────────────────────────────────────────────
    if len(vehicle_dets) > 0:
        xyxy = np.array(
            [
                [v.box[0], v.box[1], v.box[0] + v.box[2], v.box[1] + v.box[3]]
                for v in vehicle_dets
            ]
        )
        confidence = np.array([v.score for v in vehicle_dets])
        class_id = np.array([v.class_id for v in vehicle_dets])
        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
    else:
        detections = sv.Detections.empty()

    tracked_detections = tracker.update_with_detections(detections)
    active_track_ids = set()
    draw_list = []

    # Lưu lại thông tin xe cho Bước 4 (tránh phải tính toán lại x,y,w,h)
    tracked_vehicles_info = []

    # Các biến phục vụ BATCHING PARSEQ
    ocr_requests = []
    parseq_blobs = []

    # ── BƯỚC 3.1: PHA THU THẬP & CHUẨN BỊ BATCH OCR ─────────────────────────
    for i in range(len(tracked_detections)):
        xyxy = tracked_detections.xyxy[i]
        track_id = tracked_detections.tracker_id[i]
        if track_id is None:
            continue

        active_track_ids.add(track_id)
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x, y = max(0, x1), max(0, y1)
        w = min(img_bgr.shape[1] - x, x2 - x1)
        h = min(img_bgr.shape[0] - y, y2 - y1)

        if w < 60 or h < 60:
            continue

        tracked_vehicles_info.append({"track_id": track_id, "box": (x, y, w, h)})

        if track_id not in ocr_cache:
            ocr_cache[track_id] = OCRResult("", 0.0, 0)
            ocr_cache[track_id].plate_crop = None
            ocr_cache[track_id].absent_frames = 0

        ocr_cache[track_id].absent_frames = 0

        vehicle_conf = (
            float(tracked_detections.confidence[i])
            if tracked_detections.confidence is not None
            else 0.0
        )

        already_confident = (
            ocr_cache[track_id].confidence >= 0.70
            and ocr_cache[track_id].update_count >= 2
        )
        needs_ocr = (frame_id % OCR_INTERVAL == 0) and not already_confident

        if needs_ocr:
            roi_x1 = max(0, int(x - w * 0.05))
            roi_y1 = max(0, int(y - h * 0.05))
            roi_x2 = min(img_bgr.shape[1], int(x + w * 1.05))
            roi_y2 = min(img_bgr.shape[0], int(y + h * 1.05))
            vehicle_roi = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]

            if vehicle_roi.size > 0:
                plates = infer_yolo(
                    plate_session, plate_in, plate_out, vehicle_roi, 0.4
                )

                if plates:
                    plates = sorted(plates, key=lambda p: p.score, reverse=True)[:1]
                    p_det = plates[0]

                    p_x, p_y, p_w, p_h = p_det.box
                    abs_x, abs_y = p_x + roi_x1, p_y + roi_y1

                    if abs_y <= img_bgr.shape[0] * 0.9:
                        img_plate_raw = img_bgr[
                            abs_y : abs_y + p_h, abs_x : abs_x + p_w
                        ]
                        if img_plate_raw.size > 0:
                            blur_score = calculate_blur_score(img_plate_raw)
                            raw_ratio = img_plate_raw.shape[1] / (
                                img_plate_raw.shape[0] + 1e-6
                            )
                            if 0.3 <= raw_ratio <= 5.0:
                                img_plate = enhance_plate_quality(img_plate_raw)
                                if (
                                    img_plate.shape[0] / (img_plate.shape[1] + 1e-6)
                                    >= 1.5
                                ):
                                    img_plate = cv2.rotate(
                                        img_plate, cv2.ROTATE_90_CLOCKWISE
                                    )

                                ratio = img_plate.shape[1] / (img_plate.shape[0] + 1e-6)

                                if ratio > 2.2:
                                    img_plate_ready = apply_ir_handling(img_plate)
                                    blob = preprocess_and_normalize_ocr(img_plate_ready)
                                    parseq_blobs.append(blob)
                                    ocr_requests.append(
                                        {
                                            "track_id": track_id,
                                            "type": "single",
                                            "vehicle_conf": vehicle_conf,
                                            "plate_conf": p_det.score,
                                            "img_plate_raw": img_plate_raw.copy(),
                                            "blur_score": blur_score,
                                            "idx": len(parseq_blobs) - 1,
                                        }
                                    )
                                else:
                                    img_plate_ready = apply_ir_handling(img_plate)
                                    h_p, w_p = img_plate_ready.shape[:2]
                                    img_top = img_plate_ready[0 : int(h_p * 0.60), :]
                                    img_bot = img_plate_ready[int(h_p * 0.40) :, :]
                                    b_top = preprocess_and_normalize_ocr(img_top)
                                    b_bot = preprocess_and_normalize_ocr(img_bot)

                                    parseq_blobs.extend([b_top, b_bot])
                                    ocr_requests.append(
                                        {
                                            "track_id": track_id,
                                            "type": "double",
                                            "vehicle_conf": vehicle_conf,
                                            "plate_conf": p_det.score,
                                            "img_plate_raw": img_plate_raw.copy(),
                                            "blur_score": blur_score,
                                            "idx_top": len(parseq_blobs) - 2,
                                            "idx_bot": len(parseq_blobs) - 1,
                                        }
                                    )

    # ── BƯỚC 3.2: THỰC THI BATCH PARSEQ & GIẢI MÃ ───────────────────────────
    if parseq_blobs:
        # Gom tất cả blobs thành 1 tensor duy nhất có shape (N, 3, 32, 128)
        batch_input = np.concatenate(parseq_blobs, axis=0)

        # Đẩy qua GPU ĐÚNG 1 LẦN cho toàn bộ batch
        batched_logits = parseq_session.run(parseq_out, {parseq_in[0]: batch_input})[0]

        for req in ocr_requests:
            track_id = req["track_id"]
            current_text = ""
            ocr_conf = 0.0

            if req["type"] == "single":
                logits = batched_logits[req["idx"]]
                current_text, ocr_conf = decode_parseq(
                    logits, logits.shape[0], logits.shape[1]
                )
                current_text = clean_plate_text(current_text)

                holistic_score = (
                    req["plate_conf"] * WEIGHT_PLATE + ocr_conf * WEIGHT_OCR
                )
                
                if not re.match(r"^\d{2}-[A-Z][A-Z0-9]? (\d{4}|\d{3}\.\d{2})$", current_text):
                    holistic_score *= 0.70  # Phạt 30% nếu sai format (giữ nguyên logic soft-validation)
                elif len(current_text) < 6:
                    holistic_score *= 0.50  # Phạt nặng nếu quá ngắn
                    
                if len(current_text) < 6:
                    holistic_score *= 0.50
                # Nếu text giống frame trước, thưởng thêm điểm để tăng độ ổn định (tối đa +0.05)
                if current_text == ocr_cache[track_id].text:
                    bonus = min(0.05, ocr_cache[track_id].update_count * 0.02)
                    holistic_score = min(1.0, holistic_score + bonus)
            elif req["type"] == "double":
                l_top = batched_logits[req["idx_top"]]
                l_bot = batched_logits[req["idx_bot"]]

                t_top, c_top = decode_parseq(l_top, l_top.shape[0], l_top.shape[1])
                t_bot, c_bot = decode_parseq(l_bot, l_bot.shape[0], l_bot.shape[1])

                current_text = (
                    f"{clean_top_line(t_top)} {clean_bottom_line(t_bot)}".strip()
                )
                ocr_conf = (c_top + c_bot) / 2.0

                holistic_score = (
                    req["plate_conf"] * WEIGHT_PLATE + ocr_conf * WEIGHT_OCR
                )
                if not re.match(r"^\d{2}-[A-Z][A-Z0-9]? (\d{4}|\d{3}\.\d{2})$", current_text):
                    holistic_score *= 0.70

                if current_text == ocr_cache[track_id].text:
                    # Nếu text giống frame trước, thưởng thêm điểm để tăng độ ổn định (tối đa +0.05)
                    bonus = min(0.05, ocr_cache[track_id].update_count * 0.02)
                    holistic_score = min(1.0, holistic_score + bonus)

            blur = req["blur_score"]
            if blur < 80:
                holistic_score *= (
                    0.80  # Mờ nặng -> Ép giảm 20% tổng điểm
                )
            elif blur < 200:
                holistic_score *= 0.88  # Hơi nhòe -> Ép giảm 12%
            elif blur < 400:
                holistic_score *= 0.95  # Ở mức trung bình -> Ép giảm nhẹ 5%

            # Cập nhật cache nếu điểm cao hơn
            if holistic_score > ocr_cache[track_id].confidence:
                ocr_cache[track_id].text = current_text
                ocr_cache[track_id].confidence = holistic_score
                ocr_cache[track_id].update_count += 1
                ocr_cache[track_id].plate_crop = req["img_plate_raw"]

    # ── BƯỚC 4 & 6: LOGGING & VẼ BOUNDING BOX ────────────────────────────────
    for info in tracked_vehicles_info:
        track_id = info["track_id"]
        x, y, w, h = info["box"]

        res = ocr_cache.get(track_id)
        if not res:
            continue

        display_text = res.text if res.confidence >= 0.65 else ""

        if display_text and not res.is_logged:
            vehicle_crop = img_bgr[y : y + h, x : x + w]
            plate_crop = getattr(res, "plate_crop", None)

            if vehicle_crop.size > 0:
                now = datetime.now()
                t_str = now.strftime("%H:%M:%S %d/%m/%Y")
                f_ts = now.strftime("%H%M%S")
                veh_name = f"VEH_{res.text}_ID{track_id}_{f_ts}.jpg"
                pla_name = (
                    f"PLA_{res.text}_ID{track_id}_{f_ts}.jpg"
                    if (plate_crop is not None and plate_crop.size > 0)
                    else ""
                )

                log_queue.put(
                    (
                        vehicle_crop.copy(),
                        (
                            plate_crop.copy()
                            if (plate_crop is not None and plate_crop.size > 0)
                            else None
                        ),
                        t_str,
                        track_id,
                        res.text,
                        f"{res.confidence:.2f}",
                        veh_name,
                        pla_name,
                    )
                )

                res.is_logged = True

        draw_list.append(((x, y, w, h), track_id, display_text))

    # ── BƯỚC 5: XÓA CACHE ────────────────────────────────────────────────────
    for tid in list(ocr_cache.keys()):
        if tid not in active_track_ids:
            ocr_cache[tid].absent_frames = (
                getattr(ocr_cache[tid], "absent_frames", 0) + 1
            )
            if ocr_cache[tid].absent_frames > STALE_GRACE_FRAMES:
                del ocr_cache[tid]

    # Thực hiện vẽ
    for box, tid, txt in draw_list:
        dx, dy, dw, dh = box
        cv2.rectangle(img_bgr, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 2)
        label = f"ID:{tid}" + (f" | {txt}" if txt else "")
        cv2.putText(
            img_bgr,
            label,
            (dx, dy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            img_bgr,
            label,
            (dx, dy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )