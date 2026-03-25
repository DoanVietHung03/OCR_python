# pipeline.py
import cv2
import numpy as np
import supervision as sv
import os
import csv
import time
from datetime import datetime

# Import các class cấu trúc dữ liệu và tiện ích
from config import OCRResult
from utils import (enhance_plate_quality, preprocess_and_normalize_ocr, 
                   clean_plate_text, clean_top_line, clean_bottom_line)
from inference import infer_yolo, decode_parseq

# --- CẤU HÌNH LƯU TRỮ EVENT ---
LOG_DIR = "event_logs"
IMG_DIR = os.path.join(LOG_DIR, "images")
CSV_FILE = os.path.join(LOG_DIR, "history.csv")

# Tạo thư mục nếu chưa tồn tại
os.makedirs(IMG_DIR, exist_ok=True)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Thời gian", "ID_Xe", "Biển_số", "Độ_tự_tin", "File_ảnh"])

def process_single_frame(img_bgr,
                         vehicle_session, plate_session, parseq_session,
                         vehicle_in, vehicle_out,
                         plate_in, plate_out,
                         parseq_in, parseq_out,
                         allowed_vehicle_ids,
                         tracker, ocr_cache):
    """
    Xử lý trọn gói 1 khung hình: 
    Tìm xe -> Tracking -> Cắt biển -> Tăng cường ảnh -> OCR -> Lưu Log (Ảnh sạch) -> Vẽ lên ảnh hiển thị.
    """
    
    # 1. Phát hiện phương tiện bằng YOLO
    vehicle_dets = infer_yolo(vehicle_session, vehicle_in, vehicle_out, img_bgr, 0.4, allowed_vehicle_ids)

    if len(vehicle_dets) > 0:
        xyxy = np.array([[v.box[0], v.box[1], v.box[0] + v.box[2], v.box[1] + v.box[3]] for v in vehicle_dets])
        confidence = np.array([v.score for v in vehicle_dets])
        class_id = np.array([v.class_id for v in vehicle_dets])
        detections = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
    else:
        detections = sv.Detections.empty()

    # 2. Cập nhật Tracker (ByteTrack)
    tracked_detections = tracker.update_with_detections(detections)
    active_track_ids = set()

    # Danh sách tạm để chứa thông tin cần vẽ, tránh vẽ trực tiếp khi đang loop
    draw_list = []

    for i in range(len(tracked_detections)):
        xyxy = tracked_detections.xyxy[i]
        track_id = tracked_detections.tracker_id[i]
        if track_id is None: continue
            
        active_track_ids.add(track_id)
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        x, y, w, h = x1, y1, x2 - x1, y2 - y1

        # Cắt cúp an toàn trong biên giới ảnh
        x, y = max(0, x), max(0, y)
        w, h = min(img_bgr.shape[1] - x, w), min(img_bgr.shape[0] - y, h)

        if w < 60 or h < 60: continue

        final_text = ""
        needs_ocr = True

        # --- KIỂM TRA CACHE OCR ---
        if track_id in ocr_cache:
            # Nếu đã nhận diện xong (tin cậy cao hoặc đã chạy nhiều frame)
            if ocr_cache[track_id].confidence > 0.85 or ocr_cache[track_id].update_count >= 5:
                needs_ocr = False
                final_text = ocr_cache[track_id].text

        if needs_ocr:
            # ROI mở rộng 5% để không mất mép biển số
            roi_x1, roi_y1 = max(0, int(x - w * 0.05)), max(0, int(y - h * 0.05))
            roi_x2, roi_y2 = min(img_bgr.shape[1], int(x + w * 1.05)), min(img_bgr.shape[0], int(y + h * 1.05))
            vehicle_roi = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
            
            plates = infer_yolo(plate_session, plate_in, plate_out, vehicle_roi, 0.4)
            best_plate_conf = 0.0

            for p_det in plates:
                p_x, p_y, p_w, p_h = p_det.box
                abs_x, abs_y = p_x + roi_x1, p_y + roi_y1

                # Lọc biển số giả ở góc màn hình
                if abs_y > img_bgr.shape[0] * 0.9 or abs_x > img_bgr.shape[1] * 0.8: continue

                img_plate_raw = img_bgr[abs_y:abs_y+p_h, abs_x:abs_x+p_w]
                if img_plate_raw.size == 0: continue

                # Tăng cường chất lượng và xử lý xoay biển số
                img_plate = enhance_plate_quality(img_plate_raw)
                if img_plate.shape[0] / img_plate.shape[1] >= 1.5:
                    img_plate = cv2.rotate(img_plate, cv2.ROTATE_90_CLOCKWISE)

                ratio = img_plate.shape[1] / img_plate.shape[0]
                current_text, current_conf = "", 0.0

                if ratio > 2.2: # Biển dài 1 dòng
                    blob = preprocess_and_normalize_ocr(img_plate)
                    logits = parseq_session.run(parseq_out, {parseq_in[0]: blob})[0][0]
                    current_text, current_conf = decode_parseq(logits, logits.shape[0], logits.shape[1])
                    current_text = clean_plate_text(current_text)
                else: # Biển vuông 2 dòng
                    h_p, w_p = img_plate.shape[:2]
                    img_top = img_plate[0:int(h_p*0.55), :]
                    img_bot = img_plate[int(h_p*0.45):, :]

                    b_top, b_bot = preprocess_and_normalize_ocr(img_top), preprocess_and_normalize_ocr(img_bot)
                    l_top = parseq_session.run(parseq_out, {parseq_in[0]: b_top})[0][0]
                    l_bot = parseq_session.run(parseq_out, {parseq_in[0]: b_bot})[0][0]

                    t_top, c_top = decode_parseq(l_top, l_top.shape[0], l_top.shape[1])
                    t_bot, c_bot = decode_parseq(l_bot, l_bot.shape[0], l_bot.shape[1])
                    current_text = f"{clean_top_line(t_top)}-{clean_bottom_line(t_bot)}".strip('-')
                    current_conf = (c_top + c_bot) / 2.0

                if current_conf > best_plate_conf and current_conf > 0.45:
                    best_plate_conf = current_conf
                    final_text = current_text

            # --- CẬP NHẬT CACHE ---
            if final_text and final_text != "UNKNOWN":
                if track_id not in ocr_cache:
                    ocr_cache[track_id] = OCRResult(final_text, best_plate_conf, 1)
                elif best_plate_conf > ocr_cache[track_id].confidence:
                    ocr_cache[track_id].text = final_text
                    ocr_cache[track_id].confidence = best_plate_conf
                    ocr_cache[track_id].update_count += 1

        # --- LƯU LOG EVENT (ẢNH SẠCH) ---
        if track_id in ocr_cache:
            res = ocr_cache[track_id]
            # Logic lưu Log: Chỉ lưu khi tin cậy cao HOẶC đã quan sát đủ lâu
            if (res.confidence > 0.75 or res.update_count >= 3) and not res.is_logged:
                # 1. Cắt ảnh xe (crop gốc, chưa bị vẽ lên)
                vehicle_crop = img_bgr[y:y+h, x:x+w]
                
                if vehicle_crop.size > 0:
                    # 2. Tạo thông tin log
                    now = datetime.now()
                    t_str = now.strftime("%H:%M:%S %d/%m/%Y")
                    # Tên file ảnh chứa kết quả OCR và timestamp
                    f_name = f"{res.text}_ID{track_id}_{now.strftime('%H%M%S')}.jpg"
                    
                    # 3. Ghi file ảnh sạch xuống đĩa
                    cv2.imwrite(os.path.join(IMG_DIR, f_name), vehicle_crop)
                    
                    # 4. Ghi dòng dữ liệu vào CSV
                    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow([t_str, track_id, res.text, f"{res.confidence:.2f}", f_name])
                    
                    # 5. Đánh dấu đã lưu để không lưu trùng
                    res.is_logged = True

        # Thêm thông tin vào danh sách vẽ (để vẽ sau khi loop xong)
        label_text = final_text if final_text else (ocr_cache[track_id].text if track_id in ocr_cache else "Detecting...")
        draw_list.append(((x, y, w, h), track_id, label_text))

    # --- 3. VẼ LÊN FRAME HIỂN THỊ (Sau khi đã lưu ảnh event sạch) ---
    for box, tid, label in draw_list:
        dx, dy, dw, dh = box
        # Vẽ bounding box xe màu xanh lá
        cv2.rectangle(img_bgr, (dx, dy), (dx + dw, dy + dh), (0, 255, 0), 2)
        
        # Vẽ label (ID + Biển số)
        full_label = f"ID:{tid} | {label}"
        # Vẽ nét viền đen cho chữ nổi bật
        cv2.putText(img_bgr, full_label, (dx, dy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
        # Vẽ nét chính màu xanh lá
        cv2.putText(img_bgr, full_label, (dx, dy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # Dọn dẹp cache cho các ID đã biến mất
    stale_ids = [tid for tid in ocr_cache.keys() if tid not in active_track_ids]
    for tid in stale_ids: del ocr_cache[tid]