import os
import cv2
import time
import csv
import queue
import multiprocessing as mp
import onnxruntime as ort
import supervision as sv
import numpy as np
from flask import Flask, Response, render_template_string

from config import TARGET_VEHICLES
from pipeline import process_single_frame

app = Flask(__name__, static_folder="event_logs/images", static_url_path="/images")


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

    # Nâng số luồng lên để xử lý các node chạy trên CPU nhanh hơn, đặc biệt là phần post-processing của ByteTrack và Parseq
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "cudnn_conv_use_max_workspace": "1",
                "arena_extend_strategy": "kNextPowerOfTwo",
            },
        ),
        "CPUExecutionProvider",
    ]
    return (
        ort.InferenceSession("weights/yolo11s_fp16.onnx", sess_options, providers=providers),
        ort.InferenceSession(
            "weights/yolov9_detect_plate_fp16.onnx", sess_options, providers=providers
        ),
        ort.InferenceSession("weights/parseq_fp16.onnx", sess_options, providers=providers),
    )


def camera_producer(cam_name, rtsp_url, frame_queue, stop_event):
    # Khởi tạo VideoCapture với backend FFmpeg (CAP_FFMPEG)
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    # Ép OpenCV giảm buffer size xuống mức tối thiểu để tránh tích tụ delay
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while not stop_event.is_set():
        ret = cap.grab()
        if not ret:
            print(f"[{cam_name}] Mất kết nối RTSP, đang thử kết nối lại...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            continue

        ret, frame = cap.retrieve()
        if not ret:
            continue

        try:
            # Nếu queue đầy, rút frame cũ nhất ra vứt đi
            if frame_queue.full():
                frame_queue.get_nowait()

            # Đẩy frame mới nhất vào
            frame_queue.put_nowait(frame)
        except queue.Empty:
            pass
        except queue.Full:
            pass

    cap.release()


def ai_consumer(cam_name, frame_queue, display_queue, stop_event):
    vehicle_session, plate_session, parseq_session = load_models()
    vehicle_in, vehicle_out = [i.name for i in vehicle_session.get_inputs()], [
        o.name for o in vehicle_session.get_outputs()
    ]
    plate_in, plate_out = [i.name for i in plate_session.get_inputs()], [
        o.name for o in plate_session.get_outputs()
    ]
    parseq_in, parseq_out = [i.name for i in parseq_session.get_inputs()], [
        o.name for o in parseq_session.get_outputs()
    ]
    allowed_vehicle_ids = list(TARGET_VEHICLES.keys())

    # ─── BẮT ĐẦU WARM-UP ──────────────────────────────────────────
    print(f"[{cam_name}] Đang tiến hành Warm-up các model AI (sẽ mất khoảng 10-15s)...")

    # Kích thước đầu vào mặc định của YOLO là 640x640, PARSeq là 32x128
    dummy_vehicle = np.zeros((1, 3, 640, 640), dtype=np.float32)
    dummy_plate = np.zeros((1, 3, 256, 256), dtype=np.float32)
    dummy_parseq = np.zeros((1, 3, 32, 128), dtype=np.float32)

    # Chạy thử 3 lần để đảm bảo VRAM được cấp phát đầy đủ và CUDA kernel đã sẵn sàng
    for _ in range(3):
        vehicle_session.run(vehicle_out, {vehicle_in[0]: dummy_vehicle})
        plate_session.run(plate_out, {plate_in[0]: dummy_plate})
        parseq_session.run(parseq_out, {parseq_in[0]: dummy_parseq})

    print(f"[{cam_name}] Warm-up hoàn tất! Đã sẵn sàng xử lý luồng Video.")

    # XÓA RÁC TRONG QUEUE:
    # Trong lúc warm-up 15s, producer camera vẫn đẩy ảnh vào queue gây nghẽn. Cần xả hết ảnh cũ để bắt đầu với ảnh realtime mới nhất.
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except queue.Empty:
            break
    # ─── KẾT THÚC THÊM WARM-UP ─────────────────────────────────────────

    tracker = sv.ByteTrack(
        track_activation_threshold=0.5,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=25,
    )
    ocr_cache = {}
    tracker_state = {}
    frame_id = 0

    fps_start = time.time()
    fps_counter = 0
    current_fps = 0.0

    while not stop_event.is_set():
        try:
            frame = frame_queue.get_nowait()
        except queue.Empty:
            time.sleep(0.005)
            continue

        process_single_frame(
            frame,
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
            frame_id=frame_id,
        )

        cv2.putText(
            frame,
            f"CAM {cam_name} | FPS: {current_fps:.1f}",
            (20, frame.shape[0] - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (88, 255, 0),
            2,
        )
        frame_id += 1

        # Gửi tuple (Tên Camera, Frame) qua Queue để Web stream tự ghép
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        try:
            if display_queue.full():
                display_queue.get_nowait()
            display_queue.put_nowait((cam_name, small_frame))
        except:
            pass

        fps_counter += 1
        elapsed_fps = time.time() - fps_start
        if elapsed_fps >= 1.0:
            current_fps = fps_counter / elapsed_fps
            fps_counter = 0
            fps_start = time.time()


def generate_web_stream(display_queue):
    last_frames = {"K1": None, "K5": None}

    while True:
        try:
            # Nhận tuple gồm Tên Camera và Khung hình
            cam_name, frame = display_queue.get(timeout=2.0)
            last_frames[cam_name] = frame

            last_f1 = last_frames["K1"]
            last_f2 = last_frames["K5"]
            combined = None

            # Ghép ảnh dựa trên những frame mới nhất nhận được
            if last_f1 is not None and last_f2 is not None:
                h1, w1 = last_f1.shape[:2]
                h2, w2 = last_f2.shape[:2]
                f2_disp = (
                    cv2.resize(last_f2, (int(w2 * h1 / h2), h1))
                    if h1 != h2
                    else last_f2
                )
                combined = cv2.hconcat([last_f1, f2_disp])
            elif last_f1 is not None:
                combined = last_f1
            elif last_f2 is not None:
                combined = last_f2

            if combined is not None:
                ret_encode, buffer = cv2.imencode(
                    ".jpg", combined, [int(cv2.IMWRITE_JPEG_QUALITY), 65]
                )
                if ret_encode:
                    yield (
                        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                        + buffer.tobytes()
                        + b"\r\n"
                    )

        except queue.Empty:
            time.sleep(0.05)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_web_stream(app.config["DISPLAY_Q"]),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/history")
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
                    <th>Thời gian</th><th>ID Tracker</th><th>Biển số</th>
                    <th>Độ tin cậy</th><th>Ảnh Toàn Cảnh (Xe)</th><th>Ảnh Cận Cảnh (Biển số)</th>
                </tr>
            </thead>
            <tbody>
                {% for e in events %}
                <tr>
                    <td class="col-time">{{ e.get('Thời gian', '') }}</td>
                    <td class="col-id">#{{ e.get('ID_Xe', '') }}</td>
                    <td class="col-plate">{{ e.get('Biển_số', '') }}</td>
                    <td><span style="color: {% if e.get('Độ_tự_tin', '0')|float > 0.8 %}#00ff00{% else %}#ffaa00{% endif %};">{{ (e.get('Độ_tự_tin', '0')|float * 100)|int }}%</span></td>
                    <td class="img-container"><img src="/images/{{ e.get('Ảnh_Xe', '') }}" alt="Vehicle" class="img-veh"></td>
                    <td class="img-container">
                        {% if e.get('Ảnh_Biển') %}
                        <img src="/images/{{ e.get('Ảnh_Biển', '') }}" alt="Plate" class="img-pla">
                        {% else %}<span style="color: #444; font-size: 0.8rem;">(Không có ảnh)</span>{% endif %}
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </body>
    </html>
    """
    return render_template_string(html, events=events)


if __name__ == "__main__":
    rtsp_cam1 = "rtsp://localhost:8554/cong_K1"
    rtsp_cam2 = "rtsp://localhost:8554/cong_K5"

    mp.set_start_method("spawn", force=True)
    q_cam1 = mp.Queue(maxsize=2)
    q_cam2 = mp.Queue(maxsize=2)
    display_q = mp.Queue(maxsize=4)

    app.config["DISPLAY_Q"] = display_q
    stop_event = mp.Event()

    p_cam1 = mp.Process(
        target=camera_producer, args=("K1", rtsp_cam1, q_cam1, stop_event)
    )
    p_cam2 = mp.Process(
        target=camera_producer, args=("K5", rtsp_cam2, q_cam2, stop_event)
    )
    p_cam1.start()
    p_cam2.start()

    p_ai_1 = mp.Process(target=ai_consumer, args=("K1", q_cam1, display_q, stop_event))
    p_ai_2 = mp.Process(target=ai_consumer, args=("K5", q_cam2, display_q, stop_event))
    p_ai_1.start()
    p_ai_2.start()

    try:
        app.run(
            host="0.0.0.0", port=5050, debug=False, threaded=True, use_reloader=False
        )
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        p_cam1.join(timeout=2.0)
        p_cam2.join(timeout=2.0)
        p_ai_1.join(timeout=3.0)
        p_ai_2.join(timeout=3.0)
        if p_cam1.is_alive():
            p_cam1.terminate()
        if p_cam2.is_alive():
            p_cam2.terminate()
        if p_ai_1.is_alive():
            p_ai_1.terminate()
        if p_ai_2.is_alive():
            p_ai_2.terminate()