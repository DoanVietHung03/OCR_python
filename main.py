import os
import torch
import cv2
import time
import csv
import multiprocessing as mp
import onnxruntime as ort
import supervision as sv
import numpy as np
from flask import Flask, Response, render_template_string

from config import TARGET_VEHICLES
from pipeline import process_single_frame

app = Flask(__name__, static_folder='event_logs/images', static_url_path='/images')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LPR Security Dashboard - BLACK EDITION</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; background-color: #000000; color: #eeeeee; text-align: center; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
        h1 { color: #00ff88; margin-top: 10px; margin-bottom: 15px; text-shadow: 0 0 15px rgba(0, 255, 136, 0.5); font-size: 2.5rem; font-weight: 700; }
        .btn-history { display: inline-block; margin-bottom: 20px; padding: 10px 25px; background-color: #111111; color: #00ff88; text-decoration: none; font-size: 16px; font-weight: bold; border: 2px solid #00ff88; border-radius: 8px; box-shadow: 0 0 10px rgba(0, 255, 136, 0.2); transition: all 0.3s ease; }
        .btn-history:hover { background-color: #00ff88; color: #000000; box-shadow: 0 0 20px rgba(0, 255, 136, 0.6); }
        .video-container { max-width: 95%; background: #000000; padding: 10px; border-radius: 12px; border: 2px solid #00ff88; box-shadow: 0 0 25px rgba(0, 255, 136, 0.2); }
        img { width: 100%; height: auto; border-radius: 8px; display: block; }
        .status { margin-top: 25px; color: #666666; font-size: 14px; font-weight: 300; }
    </style>
</head>
<body>
    <h1>LPR SECURITY DASHBOARD</h1>
    <a href="/history" class="btn-history" target="_blank">📁 Xem Lịch Sử Nhận Diện</a>
    <div class="video-container">
        <img src="{{ url_for('video_feed') }}" alt="Live Video Feed">
    </div>
    <div class="status">● Đang stream bằng kiến trúc Multi-processing Core</div>
</body>
</html>
"""

def load_models():
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    log_severity_level = 3
    
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0, 
            'cudnn_conv_algo_search': 'HEURISTIC', 
            'cudnn_conv_use_max_workspace': '1',
            'cudnn_conv1d_pad_to_nc1d': '1',
            'arena_extend_strategy': 'kNextPowerOfTwo',
        })
    ]
    return ort.InferenceSession("weights/yolo11s.onnx", sess_options, providers=providers), \
           ort.InferenceSession("weights/yolov9_detect_plate.onnx", sess_options, providers=providers), \
           ort.InferenceSession("weights/parseq_2.onnx", sess_options, providers=providers)

def camera_worker(cam_id, video_path, shared_dict, stop_event):
    """Worker độc lập. Mỗi camera chiếm 1 Core CPU và 1 góc VRAM riêng rẽ"""
    print(f"[INFO] Process {cam_id} đang nạp AI Models vào VRAM...")
    
    # AI models CHỈ ĐƯỢC nạp bên trong Process con này để cô lập Context CUDA
    vehicle_session, plate_session, parseq_session = load_models()
    
    vehicle_in, vehicle_out = [i.name for i in vehicle_session.get_inputs()], [o.name for o in vehicle_session.get_outputs()]
    plate_in, plate_out = [i.name for i in plate_session.get_inputs()], [o.name for o in plate_session.get_outputs()]
    parseq_in, parseq_out = [i.name for i in parseq_session.get_inputs()], [o.name for o in parseq_session.get_outputs()]

    allowed_vehicle_ids = list(TARGET_VEHICLES.keys())
    
    # Tracker cũng được tạo riêng rẽ để không nhầm lẫn ID xe giữa các cổng
    tracker = sv.ByteTrack(track_activation_threshold=0.5, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=25)
    ocr_cache = {}
    tracker_state = {}

    cap = cv2.VideoCapture(video_path)
    fps_target = cap.get(cv2.CAP_PROP_FPS)
    if fps_target <= 0: fps_target = 25.0
    frame_delay = 1.0 / fps_target

    fps_timer = time.time()
    frame_count = 0
    current_fps = 0.0
    SCALE_RATIO = 0.5 
    
    empty_frame_counter = 0

    while not stop_event.is_set():
        start_time = time.time()
        cap.grab()
        ret, frame = cap.retrieve()
        
        if not ret:
            # Nếu hết video mp4, tự động phát lại từ đầu để test
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        skip_ai = False
        if empty_frame_counter > 5:
            if frame_count % 5 != 0: skip_ai = True
        else:
            if frame_count % 2 != 0: skip_ai = True

        if not skip_ai:
            process_single_frame(frame, vehicle_session, plate_session, parseq_session, 
                                vehicle_in, vehicle_out, plate_in, plate_out, parseq_in, parseq_out, 
                                allowed_vehicle_ids, tracker, ocr_cache, tracker_state, frame_id=frame_count)
            if len(tracker.tracked_tracks) == 0:
                empty_frame_counter += 1
            else:
                empty_frame_counter = 0
        else:
            if empty_frame_counter > 0: empty_frame_counter += 1

        frame_count += 1
        now = time.time()
        if now - fps_timer >= 1.0:
            current_fps = frame_count / (now - fps_timer)
            frame_count = 0
            fps_timer = now

        cv2.putText(frame, f"CAM {cam_id} | FPS: {current_fps:.1f}", (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv2.LINE_AA)
        cv2.putText(frame, f"CAM {cam_id} | FPS: {current_fps:.1f}", (20, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (88, 255, 0), 2, cv2.LINE_AA)

        dash = cv2.resize(frame, (0, 0), fx=SCALE_RATIO, fy=SCALE_RATIO)
        
        ret_encode, buffer = cv2.imencode('.jpg', dash, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if ret_encode:
            shared_dict[cam_id] = buffer.tobytes()
        
        elapsed = time.time() - start_time
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)

    cap.release()

def generate_frames(shared_dict):
    """Luồng gộp ảnh (Aggregator) chỉ có nhiệm vụ đọc từ Shared Memory"""
    while True:
        buf1 = shared_dict.get('K1')
        buf2 = shared_dict.get('K5')
        
        # BÍ QUYẾT TĂNG TỐC 2: Giải mã Byte ngược lại thành ảnh
        frame1 = cv2.imdecode(np.frombuffer(buf1, np.uint8), cv2.IMREAD_COLOR) if buf1 is not None else None
        frame2 = cv2.imdecode(np.frombuffer(buf2, np.uint8), cv2.IMREAD_COLOR) if buf2 is not None else None
        
        combined = None
        if frame1 is not None and frame2 is not None:
            h1, w1 = frame1.shape[:2]
            h2, w2 = frame2.shape[:2]
            if h1 != h2: frame2 = cv2.resize(frame2, (int(w2 * h1 / h2), h1))
            combined = cv2.hconcat([frame1, frame2])
        elif frame1 is not None: combined = frame1
        elif frame2 is not None: combined = frame2
        
        if combined is not None:
            ret, final_buffer = cv2.imencode('.jpg', combined, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + final_buffer.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # Giới hạn tốc độ gộp ảnh ở khoảng 30 FPS

# --- Các Route của Flask ---
@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/history')
def view_history():
    # Giữ nguyên toàn bộ mã trang Lịch sử của bạn ở đây
    events = []
    csv_path = "event_logs/history.csv"
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            events = list(reader)
    events.reverse() 
    
    html = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="UTF-8">
        <title>Lịch sử Nhận diện LPR</title>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #0a0a0a; color: #eee; padding: 20px; }
            .header-nav { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; border-bottom: 2px solid #00ff88; padding-bottom: 10px; }
            .btn-back { color: #00ff88; text-decoration: none; font-weight: bold; border: 1px solid #00ff88; padding: 5px 15px; border-radius: 5px; }
            .btn-back:hover { background: #00ff88; color: #000; }
            table { width: 100%; border-collapse: collapse; background: #111; box-shadow: 0 0 20px rgba(0,0,0,0.5); }
            th, td { border: 1px solid #222; padding: 15px; text-align: center; }
            th { background: #1a1a1a; color: #00ff88; text-transform: uppercase; letter-spacing: 1px; font-size: 0.9rem;}
            .col-time { color: #888; font-family: monospace; font-size: 1.1rem; }
            .col-plate { font-size: 1.5rem; font-weight: bold; color: #ffff00; text-shadow: 0 0 5px rgba(255,255,0,0.3); }
            .col-id { color: #00e5ff; font-weight: bold; }
            .img-container img { border-radius: 4px; transition: transform 0.2s; }
            .img-veh { max-width: 200px; border: 1px solid #333; }
            .img-veh:hover { transform: scale(2.5); z-index: 100; position: relative; border-color: #00ff88; }
            .img-pla { max-width: 150px; border: 2px solid #00e5ff; }
            .img-pla:hover { transform: scale(3.5); z-index: 100; position: relative; box-shadow: 0 0 15px #00e5ff;}
            tr:hover { background: #181818; }
        </style>
    </head>
    <body>
        <div class="header-nav">
            <h2>LỊCH SỬ HỆ THỐNG LPR</h2>
            <a href="/" class="btn-back">&larr; QUAY LẠI LIVE CAMERA</a>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Thời gian</th>
                    <th>ID Tracker</th>
                    <th>Biển số</th>
                    <th>Độ tin cậy</th>
                    <th>Ảnh Toàn Cảnh (Xe)</th>
                    <th>Ảnh Cận Cảnh (Biển số)</th>
                </tr>
            </thead>
            <tbody>
                {% for e in events %}
                <tr>
                    <td class="col-time">{{ e.get('Thời gian', '') }}</td>
                    <td class="col-id">#{{ e.get('ID_Xe', '') }}</td>
                    <td class="col-plate">{{ e.get('Biển_số', '') }}</td>
                    <td><span style="color: {% if e.get('Độ_tự_tin', '0')|float > 0.8 %}#00ff00{% else %}#ffaa00{% endif %};">{{ (e.get('Độ_tự_tin', '0')|float * 100)|int }}%</span></td>
                    <td class="img-container">
                        <img src="/images/{{ e.get('Ảnh_Xe', '') }}" alt="Vehicle" class="img-veh">
                    </td>
                    <td class="img-container">
                        {% if e.get('Ảnh_Biển') %}
                        <img src="/images/{{ e.get('Ảnh_Biển', '') }}" alt="Plate" class="img-pla">
                        {% else %}
                        <span style="color: #444; font-size: 0.8rem;">(Không có ảnh)</span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """
    return render_template_string(html, events=events)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(app.config['SHARED_DICT']), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Rất quan trọng: Ép Linux dùng 'spawn' thay vì 'fork' để tránh vỡ Context CUDA
    mp.set_start_method('spawn', force=True)
    
    manager = mp.Manager()
    shared_display = manager.dict()
    shared_display['K1'] = None
    shared_display['K5'] = None
    
    # Gắn shared_display vào config của Flask để truyền qua Route
    app.config['SHARED_DICT'] = shared_display
    stop_event = mp.Event()

    print("[INFO] Khởi động kiến trúc Đa tiến trình (Multi-processing)...")
    
    # Khởi tạo tiến trình độc lập cho từng camera
    p1 = mp.Process(target=camera_worker, args=('K1', "CCTV/cong_K1.mp4", shared_display, stop_event))
    p2 = mp.Process(target=camera_worker, args=('K5', "CCTV/cong_K5.mp4", shared_display, stop_event))
    
    p1.start()
    p2.start()

    print("[INFO] Server đang chạy! Mở trình duyệt và truy cập: http://<ĐỊA_CHỈ_IP_UBUNTU>:5050")
    try:
        # Tắt reloader để tránh Flask vô tình đẻ thêm tiến trình (spawn) 2 lần
        app.run(host='0.0.0.0', port=5050, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("[INFO] Đang tắt hệ thống một cách an toàn...")
    finally:
        stop_event.set()
        p1.join(timeout=2.0)
        p2.join(timeout=2.0)