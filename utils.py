import cv2
import numpy as np

DICT_NUM_TO_CHAR = {"0": "D", "1": "I", "2": "Z", "5": "S", "8": "B"}
DICT_CHAR_TO_NUM = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "B": "8",
    "S": "5",
    "G": "6",
}


def clean_plate_text(input_str):
    """Xử lý biển 1 dòng (Ô tô / Rơ moóc): Tổng hợp format của cả 2 dòng trên"""
    raw = "".join([c for c in input_str.upper() if c.isalnum()])
    if len(raw) < 6:
        return raw

    # Phân tách: 2 ký tự đầu (số) + 1 ký tự (chữ) + 1 ký tự (số/chữ) + phần còn lại (số)
    p1 = "".join([DICT_CHAR_TO_NUM.get(c, c) for c in raw[:2]])
    p2 = DICT_NUM_TO_CHAR.get(raw[2], raw[2])
    p3 = raw[3]  # Ký tự thứ 4 linh động

    # Ép phần đuôi (4 hoặc 5 ký tự cuối) thành số
    p4_raw = raw[4:]
    p4 = "".join([DICT_CHAR_TO_NUM.get(c, c) for c in p4_raw])

    head = p1 + "-" + p2 + p3
    tail = p4
    if len(tail) >= 5:
        tail = f"{tail[:3]}.{tail[3:5]}"

    return f"{head} {tail}".strip()


def clean_bottom_line(input_str):
    corrected = "".join([DICT_CHAR_TO_NUM.get(c, c) for c in input_str.upper()])
    output = "".join([c for c in input_str if c.isdigit()])
    if len(output) >= 5:  # Biển 5 số (có thể có nhiễu dư phía sau)
        return f"{output[:3]}.{output[3:5]}"
    elif len(output) == 4:  # Biển 4 số đời cũ
        return output
    return output  # Trả về nguyên gốc nếu không xác định được


def clean_top_line(input_str):
    raw = "".join([c for c in input_str.upper() if c.isalnum()])
    if len(raw) < 3:
        return raw
    part1 = "".join([DICT_CHAR_TO_NUM.get(c, c) for c in raw[:2]])
    part2 = DICT_NUM_TO_CHAR.get(raw[2], raw[2])
    part3 = raw[3:]
    output = part1 + part2 + part3
    output = output[:4]
    if len(output) >= 3:
        return f"{output[:2]}-{output[2:]}"
    return output


def preprocess_and_normalize_ocr(src, target_w=128, target_h=32):
    if src is None or src.size == 0:
        return None
    return cv2.dnn.blobFromImage(
        src,
        scalefactor=1.0 / 127.5,
        size=(target_w, target_h),
        mean=(127.5, 127.5, 127.5),
        swapRB=True,
        crop=False,
    )


def enhance_plate_quality(src):
    if src is None or src.size == 0:
        return src

    # Lọc nhiễu TRƯỚC khi phóng to giúp tiết kiệm 70% CPU time cho hàm này
    dst = cv2.bilateralFilter(src, d=5, sigmaColor=25, sigmaSpace=25)

    # Phóng to ảnh sau khi đã lọc nhiễu
    dst = cv2.resize(dst, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(dst, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    avg_brightness = np.mean(l)

    # 3. Tăng cường độ sáng/tối (CLAHE)
    if avg_brightness < 70:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 4))
    elif avg_brightness > 200:
        clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 4))
    else:
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 4))

    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))

    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def calculate_blur_score(img):
    """
    Tính toán độ mờ của ảnh bằng Laplacian Variance.
    Điểm càng cao -> Ảnh càng nét.
    Điểm càng thấp -> Ảnh càng mờ/nhòe.
    """
    if img is None or img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def letterbox_yolo(source, expected_width, expected_height):
    h, w = source.shape[:2]
    ratio = min(expected_height / h, expected_width / w)
    new_unpad_w = int(round(w * ratio))
    new_unpad_h = int(round(h * ratio))

    resized = cv2.resize(source, (new_unpad_w, new_unpad_h))
    pad_w = (expected_width - new_unpad_w) // 2
    pad_h = (expected_height - new_unpad_h) // 2

    padded = np.full((expected_height, expected_width, 3), 114, dtype=np.uint8)
    padded[pad_h : pad_h + new_unpad_h, pad_w : pad_w + new_unpad_w] = resized
    return padded, ratio, pad_w, pad_h


def is_ir_image(img, saturation_thresh=15):
    """
    Phát hiện ảnh hồng ngoại dựa vào kênh Saturation.
    """
    if img is None or img.size == 0:
        return False
    # Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]

    # Tính trung bình độ bão hòa màu
    mean_s = np.mean(s_channel)

    # Nếu trung bình S nhỏ hơn ngưỡng (ví dụ 15), đó là ảnh hồng ngoại
    return mean_s < saturation_thresh


def apply_ir_handling(img_plate):
    """
    Đảo ngược cực (Invert) nếu biển số IR bị lóa phản quang.
    """
    if not is_ir_image(img_plate):
        return img_plate  # Nếu là ảnh ban ngày có màu, giữ nguyên

    # Chuyển sang ảnh xám để phân tích độ sáng
    gray = cv2.cvtColor(img_plate, cv2.COLOR_BGR2GRAY)

    # Trích xuất vùng trung tâm của biển số (bỏ qua viền)
    h, w = gray.shape
    margin_y, margin_x = int(h * 0.25), int(w * 0.25)
    center_roi = gray[margin_y : h - margin_y, margin_x : w - margin_x]

    if center_roi.size > 0:
        mean_center_brightness = np.mean(center_roi)
        # Biển số phản quang ban đêm thường có chữ lóa rất sáng
        # Nếu trung tâm quá sáng (> 140), khả năng cao chữ đang trắng trên nền đen
        if mean_center_brightness > 140:
            # Lật ngược pixel (chữ trắng -> đen, nền đen -> trắng)
            img_plate = cv2.bitwise_not(img_plate)

    return img_plate