# main.py
import cv2
import time
import os
import csv
import threading
import multiprocessing
import onnxruntime as ort
import supervision as sv
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, Response, render_template_string

from config import TARGET_VEHICLES
from pipeline import process_single_frame

app = Flask(__name__, static_folder='event_logs/images', static_url_path='/images')

# --- THAY ĐỔI LỚN: Dùng biến toàn cục để chống giật lag thay vì Queue ---
latest_display = {"combined": None}
display_lock = threading.Lock()
stop_event = threading.Event()

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
    <div class="status">● Đang stream trực tiếp từ Ubuntu Server (Headless)</div>
</body>
</html>
"""

class ThreadedCamera:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps <= 0: self.fps = 25.0 
            
        self.target_frame_time = 1.0 / self.fps
        self.ret, self.frame = self.cap.read()
        self.has_new = True 
        self.running = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            start_time = time.time() 
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            
            self.frame = frame 
            self.ret = ret
            self.has_new = True 
            
            elapsed_time = time.time() - start_time
            sleep_time = self.target_frame_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self):
        new = self.has_new
        self.has_new = False
        return self.ret, self.frame, new
        
    def isOpened(self): return self.cap.isOpened()
    def release(self):
        self.running = False
        if self.thread.is_alive(): self.thread.join()
        self.cap.release()

def video_processing_worker(stop_event, vehicle_session, plate_session, parseq_session):
    vehicle_in, vehicle_out = [i.name for i in vehicle_session.get_inputs()], [o.name for o in vehicle_session.get_outputs()]
    plate_in, plate_out = [i.name for i in plate_session.get_inputs()], [o.name for o in plate_session.get_outputs()]
    parseq_in, parseq_out = [i.name for i in parseq_session.get_inputs()], [o.name for o in parseq_session.get_outputs()]

    allowed_vehicle_ids = list(TARGET_VEHICLES.keys())

    cap1 = ThreadedCamera("CCTV/cong_K1.mp4")
    cap2 = ThreadedCamera("CCTV/cong_K5.mp4")

    if not cap1.isOpened() or not cap2.isOpened():
        print("[LỖI] Không thể đọc video. Kiểm tra lại đường dẫn file!")
        return

    tracker_cam1 = sv.ByteTrack(track_activation_threshold=0.5, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=25)
    tracker_cam2 = sv.ByteTrack(track_activation_threshold=0.5, lost_track_buffer=30, minimum_matching_threshold=0.8, frame_rate=25)

    cache_cam1, cache_cam2 = {}, {}
    executor = ThreadPoolExecutor(max_workers=2)
    SCALE_RATIO = 0.6 

    fps_timer = time.time()
    cam1_frame_count = 0
    cam2_frame_count = 0
    cam1_fps = 0.0
    cam2_fps = 0.0
    
    last_display1, last_display2 = None, None

    try:
        while not stop_event.is_set():
            ret1, frame1, new1 = cap1.read()
            ret2, frame2, new2 = cap2.read()

            if not ret1 and not ret2:
                print("[INFO] Video đã kết thúc.")
                break 

            if not new1 and not new2:
                time.sleep(0.005)
                continue

            # Tính tổng số frame đã xử lý trong 1 giây
            if ret1 and new1: cam1_frame_count += 1
            if ret2 and new2: cam2_frame_count += 1
            
            now = time.time()
            if now - fps_timer >= 1.0:
                cam1_fps = cam1_frame_count / (now - fps_timer)
                cam2_fps = cam2_frame_count / (now - fps_timer)
                cam1_frame_count = 0
                cam2_frame_count = 0
                fps_timer = now

            future1, future2 = None, None

            if ret1 and new1:
                disp1 = frame1.copy()
                future1 = executor.submit(process_single_frame, disp1, vehicle_session, plate_session, parseq_session, vehicle_in, vehicle_out, plate_in, plate_out, parseq_in, parseq_out, allowed_vehicle_ids, tracker_cam1, cache_cam1)
            
            if ret2 and new2:
                disp2 = frame2.copy()
                future2 = executor.submit(process_single_frame, disp2, vehicle_session, plate_session, parseq_session, vehicle_in, vehicle_out, plate_in, plate_out, parseq_in, parseq_out, allowed_vehicle_ids, tracker_cam2, cache_cam2)

            if future1: 
                future1.result()
                last_display1 = disp1
            if future2: 
                future2.result()
                last_display2 = disp2

            dash1, dash2 = None, None
            if ret1 and last_display1 is not None:
                d1 = last_display1.copy()
                cv2.putText(d1, f"CAM 1 | FPS: {cam1_fps:.1f}", (20, d1.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(d1, f"CAM 1 | FPS: {cam1_fps:.1f}", (20, d1.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (88, 255, 0), 2, cv2.LINE_AA)
                dash1 = cv2.resize(d1, (0, 0), fx=SCALE_RATIO, fy=SCALE_RATIO)

            if ret2 and last_display2 is not None:
                d2 = last_display2.copy()
                cv2.putText(d2, f"CAM 2 | FPS: {cam2_fps:.1f}", (20, d2.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 4, cv2.LINE_AA)
                cv2.putText(d2, f"CAM 2 | FPS: {cam2_fps:.1f}", (20, d2.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (88, 255, 0), 2, cv2.LINE_AA)
                dash2 = cv2.resize(d2, (0, 0), fx=SCALE_RATIO, fy=SCALE_RATIO)
                
            # Ghép ảnh và lưu vào biến toàn cục (Global Lock)
            combined = None
            if dash1 is not None and dash2 is not None:
                h1, w1 = dash1.shape[:2]
                h2, w2 = dash2.shape[:2]
                if h1 != h2: dash2 = cv2.resize(dash2, (int(w2 * h1 / h2), h1))
                combined = cv2.hconcat([dash1, dash2])
            elif dash1 is not None: combined = dash1
            elif dash2 is not None: combined = dash2
            
            if combined is not None:
                with display_lock:
                    latest_display["combined"] = combined

    finally:
        executor.shutdown(wait=False)
        cap1.release()
        cap2.release()

def load_models():
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = max(2, int(multiprocessing.cpu_count() * 0.75))

    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0, 
            'cudnn_conv_algo_search': 'EXHAUSTIVE', 
            'arena_extend_strategy': 'kNextPowerOfTwo', 
            'do_copy_in_default_stream': False
        }),
        'CPUExecutionProvider'
    ]
    return ort.InferenceSession("weights/yolo11s.onnx", sess_options, providers=providers), ort.InferenceSession("weights/yolov9_detect_plate.onnx", sess_options, providers=providers), ort.InferenceSession("weights/parseq_2.onnx", sess_options, providers=providers)

def generate_frames():
    """Hàm lấy ảnh độc lập cho Web, ổn định ở mức 30 FPS"""
    while True:
        frame_to_encode = None
        with display_lock:
            if latest_display["combined"] is not None:
                frame_to_encode = latest_display["combined"].copy()
        
        if frame_to_encode is not None:
            ret, buffer = cv2.imencode('.jpg', frame_to_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        # Ép tốc độ mạng stream ở khoảng 30 FPS để không làm nặng trình duyệt
        time.sleep(0.033)

# --- Các Route của Flask ---
@app.route('/')
def index(): return render_template_string(HTML_TEMPLATE)

@app.route('/history')
def view_history():
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
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("[INFO] Đang tải các mô hình ONNX...")
    vehicle_session, plate_session, parseq_session = load_models()
    
    print("[INFO] Khởi động luồng xử lý AI (Background Thread)...")
    worker_thread = threading.Thread(
        target=video_processing_worker,
        args=(stop_event, vehicle_session, plate_session, parseq_session),
        daemon=True
    )
    worker_thread.start()

    print("[INFO] Server đang chạy! Mở trình duyệt và truy cập: http://<ĐỊA_CHỈ_IP_UBUNTU>:5050")
    try:
        app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("[INFO] Đang tắt server...")
    finally:
        stop_event.set()
        worker_thread.join(timeout=2.0)