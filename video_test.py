import cv2
import time
import numpy as np
import onnxruntime as ort
import supervision as sv

from config import TARGET_VEHICLES
from pipeline import process_single_frame


def load_models():
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 4

    providers = [
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "cudnn_conv_algo_search": "HEURISTIC",
                "cudnn_conv_use_max_workspace": "1",
                "arena_extend_strategy": "kNextPowerOfTwo",
            },
        ),
        "CPUExecutionProvider",
    ]
    return (
        ort.InferenceSession("weights/yolo11s.onnx", sess_options, providers=providers),
        ort.InferenceSession(
            "weights/yolov9_detect_plate.onnx", sess_options, providers=providers
        ),
        ort.InferenceSession("weights/parseq.onnx", sess_options, providers=providers),
    )


def process_video(input_path, output_path):
    print("⏳ Đang nạp các model AI vào GPU...")
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

    print(f"🎬 Đang mở video: {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Lỗi: Không thể mở được file video {input_path}")
        return

    # Lấy thông số của video gốc để setup VideoWriter
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Khởi tạo công cụ ghi video (VideoWriter)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec chuẩn cho đuôi .mp4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = sv.ByteTrack(
        track_activation_threshold=0.5,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=fps,
    )
    ocr_cache = {}
    tracker_state = {}
    frame_id = 0

    print(
        f"⚙️ Bắt đầu xử lý! Tổng số frame: {total_frames} | Độ phân giải: {width}x{height} | FPS gốc: {fps}"
    )
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Hết video

        # Đo tốc độ xử lý thực tế của GPU
        frame_start_time = time.time()

        # Đẩy vào Pipeline xử lý (vẽ trực tiếp lên biến `frame`)
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

        frame_end_time = time.time()
        process_fps = 1.0 / (frame_end_time - frame_start_time + 1e-6)

        # In thêm thông số tiến độ lên góc trên cùng của khung hình
        cv2.putText(
            frame,
            f"GPU FPS: {process_fps:.1f} | Frame: {frame_id}/{total_frames}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
        )

        # Ghi frame đã được vẽ bounding box vào file video mới
        out.write(frame)
        frame_id += 1

        # Cập nhật tiến độ lên terminal sau mỗi 30 frame
        if frame_id % 30 == 0:
            percent = (frame_id / total_frames) * 100
            print(
                f" ➜ Đã xử lý: {frame_id}/{total_frames} frames ({percent:.1f}%) - Tốc độ: {process_fps:.1f} FPS"
            )

    # Dọn dẹp bộ nhớ và đóng file
    cap.release()
    out.release()
    total_time = time.time() - start_time

    print("=" * 50)
    print(f"✅ HOÀN TẤT! Video đã được lưu tại: {output_path}")
    print(f"⏱️ Tổng thời gian xử lý: {total_time:.1f} giây")
    print(f"🚀 Tốc độ trung bình: {total_frames / total_time:.1f} FPS")
    print("=" * 50)


if __name__ == "__main__":
    # Thay đổi tên file video đầu vào và đầu ra tại đây
    INPUT_VIDEO = "CCTV/cong_K5.mp4"
    OUTPUT_VIDEO = "result_cong_K5.mp4"

    process_video(INPUT_VIDEO, OUTPUT_VIDEO)