import cv2
import numpy as np


def clean_plate_text(input_str):
    return "".join([c.upper() for c in input_str if c.isalnum()])


def clean_bottom_line(input_str):
    output = "".join([c for c in input_str if c.isdigit()])
    if len(output) >= 5:  # Biển 5 số (có thể có nhiễu dư phía sau)
        return f"{output[:3]}.{output[3:5]}"
    elif len(output) == 4:  # Biển 4 số đời cũ
        return output
    return output  # Trả về nguyên gốc nếu không xác định được


def clean_top_line(input_str):
    output = clean_plate_text(input_str)[:4]
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