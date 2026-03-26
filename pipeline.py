# pipeline.py
import cv2
import numpy as np
import supervision as sv
import os
import csv
import time
from datetime import datetime
import re

from config import OCRResult, WEIGHT_VEHICLE, WEIGHT_PLATE, WEIGHT_OCR
from utils import (enhance_plate_quality, preprocess_and_normalize_ocr, 
                   clean_plate_text, clean_top_line, clean_bottom_line)
from inference import infer_yolo, decode_parseq

LOG_DIR = "event_logs"
IMG_DIR = os.path.join(LOG_DIR, "images")
CSV_FILE = os.path.join(LOG_DIR, "history.csv")

os.makedirs(IMG_DIR, exist_ok=True)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Thêm cột Ảnh_Biển vào file CSV
        writer.writerow(["Thời gian", "ID_Xe", "Biển_số", "Độ_tự_tin", "Ảnh_Xe", "Ảnh_Biển"])

def process_single_frame(img_bgr,
                         vehicle_session, plate_session, parseq_session,
                         vehicle_in, vehicle_out,
                         plate_in, plate_out,
                         parseq_in, parseq_out,
                         allowed_vehicle_ids,
                         tracker, ocr_cache, frame_id=0):
    
    vehicle_dets = infer_yolo(vehicle_session, vehicle_in, vehicle_out, img_bgr, 0.4, allowed_vehicle_ids)

    if len(vehicle_dets) > 0:
        xyxy = np.array([[v.box[0], v.box[1], v.box[0] + v.box[2], v.box[1] + v.box[3]] for v in vehicle_dets])
        confidence = np.array([v.score for v in vehicle_dets])
        class_id = np.array([v.class_id for v in vehicle_dets])
        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
    else:
        detections = sv.Detections.empty()

    tracked_detections = tracker.update_with_detections(detections)
    active_track_ids = set()
    draw_list = []

    for i in range(len(tracked_detections)):
        xyxy = tracked_detections.xyxy[i]
        track_id = tracked_detections.tracker_id[i]
        if track_id is None: continue
            
        active_track_ids.add(track_id)
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x, y, w, h = max(0, x1), max(0, y1), x2 - x1, y2 - y1
        w, h = min(img_bgr.shape[1] - x, w), min(img_bgr.shape[0] - y, h)

        if w < 60 or h < 60: continue

        final_text = ""
        needs_ocr = True

        if track_id in ocr_cache:
            if ocr_cache[track_id].confidence >= 0.65 and ocr_cache[track_id].update_count >= 5:
                needs_ocr = False
                final_text = ocr_cache[track_id].text

        vehicle_conf = tracked_detections.confidence[i] if tracked_detections.confidence is not None else 0.0
        
        if needs_ocr and (frame_id % 3 == 0):
            roi_x1, roi_y1 = max(0, int(x - w * 0.05)), max(0, int(y - h * 0.05))
            roi_x2, roi_y2 = min(img_bgr.shape[1], int(x + w * 1.05)), min(img_bgr.shape[0], int(y + h * 1.05))
            vehicle_roi = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            
            plates = infer_yolo(plate_session, plate_in, plate_out, vehicle_roi, 0.4)
            best_plate_conf = 0.0
            best_plate_img = None # Biến lưu tạm ảnh crop biển số tốt nhất

            for p_det in plates:
                p_x, p_y, p_w, p_h = p_det.box
                plate_conf = p_det.score
                
                abs_x, abs_y = p_x + roi_x1, p_y + roi_y1
                if abs_y > img_bgr.shape[0] * 0.9: continue

                img_plate_raw = img_bgr[abs_y:abs_y+p_h, abs_x:abs_x+p_w]
                if img_plate_raw.size == 0: continue
                
                raw_ratio = img_plate_raw.shape[1] / img_plate_raw.shape[0]
                if raw_ratio < 0.3 or raw_ratio > 5.0:
                    continue

                img_plate = enhance_plate_quality(img_plate_raw)
                if img_plate.shape[0] / img_plate.shape[1] >= 1.5:
                    img_plate = cv2.rotate(img_plate, cv2.ROTATE_90_CLOCKWISE)

                ratio = img_plate.shape[1] / img_plate.shape[0]
                current_text, ocr_conf = "", 0.0

                if ratio > 2.2:
                    blob = preprocess_and_normalize_ocr(img_plate)
                    logits = parseq_session.run(parseq_out, {parseq_in[0]: blob})[0][0]
                    current_text, ocr_conf = decode_parseq(logits, logits.shape[0], logits.shape[1])
                    current_text = clean_plate_text(current_text)
                else:
                    h_p, w_p = img_plate.shape[:2]
                    img_top, img_bot = img_plate[0:int(h_p*0.60), :], img_plate[int(h_p*0.40):, :]
                    
                    b_top, b_bot = preprocess_and_normalize_ocr(img_top), preprocess_and_normalize_ocr(img_bot)
                    l_top = parseq_session.run(parseq_out, {parseq_in[0]: b_top})[0][0]
                    l_bot = parseq_session.run(parseq_out, {parseq_in[0]: b_bot})[0][0]
                    
                    t_top, c_top = decode_parseq(l_top, l_top.shape[0], l_top.shape[1])
                    t_bot, c_bot = decode_parseq(l_bot, l_bot.shape[0], l_bot.shape[1])
                    
                    current_text = f"{clean_top_line(t_top)} {clean_bottom_line(t_bot)}".strip()
                    ocr_conf = (c_top + c_bot) / 2.0

                holistic_score = (vehicle_conf * WEIGHT_VEHICLE) + (plate_conf * WEIGHT_PLATE) + (ocr_conf * WEIGHT_OCR)
                
                # --- Kiểm tra định dạng bằng regex ---
                if ratio > 2.2:
                    # Nếu là biển dài (ô tô), tối thiểu phải có 6 ký tự. "AU1" sẽ bị phạt.
                    if len(current_text) < 6:
                        holistic_score *= 0.50
                else:
                    # Nếu là biển ngắn (xe máy), phải đúng định dạng "XX-YX 123.45". Nếu không, phạt nặng.
                    pattern = r"^\d{2}-[A-Z]\d \d{3}\.\d{2}$"
                    if not re.match(pattern, current_text):
                        holistic_score *= 0.70
                
                if holistic_score > best_plate_conf:
                    best_plate_conf = holistic_score
                    final_text = current_text
                    # Copy ảnh biển số gốc vào biến tạm
                    best_plate_img = img_plate_raw.copy()

            if final_text:
                if track_id not in ocr_cache:
                    ocr_cache[track_id] = OCRResult(final_text, best_plate_conf, 1)
                    ocr_cache[track_id].plate_crop = best_plate_img # Gắn ảnh biển vào bộ nhớ
                elif best_plate_conf > ocr_cache[track_id].confidence:
                    ocr_cache[track_id].text = final_text
                    ocr_cache[track_id].confidence = best_plate_conf
                    ocr_cache[track_id].update_count += 1
                    ocr_cache[track_id].plate_crop = best_plate_img # Cập nhật ảnh biển nét hơn

        display_text = ""
        if track_id in ocr_cache:
            res = ocr_cache[track_id]
            if res.confidence >= 0.65:
                display_text = res.text
                
                if not res.is_logged:
                    vehicle_crop = img_bgr[y:y+h, x:x+w]
                    # Lấy ảnh biển số từ bộ nhớ ra
                    plate_crop = getattr(res, 'plate_crop', None) 
                    
                    if vehicle_crop.size > 0:
                        now = datetime.now()
                        t_str, f_ts = now.strftime("%H:%M:%S %d/%m/%Y"), now.strftime("%H%M%S")
                        
                        # Tạo 2 tên file tách biệt
                        veh_name = f"VEH_{res.text}_ID{track_id}_{f_ts}.jpg"
                        pla_name = f"PLA_{res.text}_ID{track_id}_{f_ts}.jpg"
                        
                        cv2.imwrite(os.path.join(IMG_DIR, veh_name), vehicle_crop)
                        
                        # Chỉ lưu nếu cắt được ảnh biển số
                        if plate_crop is not None and plate_crop.size > 0:
                            cv2.imwrite(os.path.join(IMG_DIR, pla_name), plate_crop)
                        else:
                            pla_name = "" # Để trống nếu ko có
                            
                        with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            # Ghi cả 2 tên file vào dòng mới
                            writer.writerow([t_str, track_id, res.text, f"{res.confidence:.2f}", veh_name, pla_name])
                        res.is_logged = True

        draw_list.append(((x, y, w, h), track_id, display_text))

    for box, tid, txt in draw_list:
        dx, dy, dw, dh = box
        cv2.rectangle(img_bgr, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 2)
        label = f"ID:{tid}" + (f" | {txt}" if txt else "")
        cv2.putText(img_bgr, label, (dx, dy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img_bgr, label, (dx, dy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    stale_ids = [tid for tid in ocr_cache.keys() if tid not in active_track_ids]
    for tid in stale_ids: del ocr_cache[tid]