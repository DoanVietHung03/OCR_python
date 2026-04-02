"""
prepare_test_data.py
====================
Chuyển đổi dataset Roboflow (YOLO format, có plate_text trong tên class)
thành cấu trúc thư mục test_data/ dùng được với benchmark_alpr.py

Dataset Roboflow OCR thường có label dạng:
  0 0.512 0.431 0.231 0.089   ← class=0 là "character", không có text
  
→ Cần bổ sung plate_text từ tên file hoặc annotation khác.

Script này cũng tự crop biển số từ ảnh gốc ra thư mục crops/
để chạy OCR-only benchmark.
"""

import os
import cv2
import shutil
import argparse
from pathlib import Path


def prepare_from_roboflow_yolo(
    src_dir: str,           # thư mục chứa images/ và labels/ gốc
    dst_dir: str = "test_data",
    split:   str = "test",  # "train", "valid", "test"
):
    """
    Cấu trúc Roboflow:
      src_dir/
        test/
          images/*.jpg
          labels/*.txt   (YOLO bbox, class = ký tự từng cái hoặc plate_text)
    """
    src = Path(src_dir) / split
    dst = Path(dst_dir)
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    (dst / "crops").mkdir(parents=True, exist_ok=True)

    img_dir = src / "images"
    lbl_dir = src / "labels"

    count = 0
    for img_path in sorted(img_dir.glob("*.[jp][pn]g")):
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        lines = lbl_path.read_text().strip().splitlines()
        if not lines:
            continue

        # ── Case 1: label đã có plate_text ở cột cuối ────────────────────
        # Format: class cx cy w h plate_text
        # → Copy thẳng sang dst/labels/
        if len(lines[0].split()) >= 6:
            shutil.copy(img_path, dst / "images" / img_path.name)
            shutil.copy(lbl_path, dst / "labels" / lbl_path.name)

            # Crop biển số ra thư mục crops/
            for i, line in enumerate(lines):
                parts = line.strip().split()
                cx, cy, w, h = map(float, parts[1:5])
                plate_text   = parts[5]
                x1 = int((cx - w/2) * W)
                y1 = int((cy - h/2) * H)
                x2 = int((cx + w/2) * W)
                y2 = int((cy + h/2) * H)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crop_name = f"{img_path.stem}_{i}.jpg"
                cv2.imwrite(str(dst / "crops" / crop_name), crop)
                (dst / "crops" / f"{img_path.stem}_{i}.txt").write_text(plate_text)
            count += 1

        # ── Case 2: label Roboflow OCR kiểu character bbox ───────────────
        # Format: char_class cx cy w h  (class = index của ký tự)
        # → Cần classes.txt để map index → ký tự, rồi ghép lại thành plate_text
        # TODO: implement nếu dataset của bạn dùng format này
        else:
            print(f"[Skip] {img_path.name} — label không có plate_text")

    print(f"[Done] Prepared {count} samples → {dst_dir}/")
    print(f"  images/  : {len(list((dst/'images').glob('*')))}")
    print(f"  labels/  : {len(list((dst/'labels').glob('*.txt')))}")
    print(f"  crops/   : {len(list((dst/'crops').glob('*.jpg')))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",   required=True, help="Thư mục Roboflow dataset gốc")
    parser.add_argument("--dst",   default="test_data", help="Thư mục output")
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    args = parser.parse_args()

    prepare_from_roboflow_yolo(args.src, args.dst, args.split)

    print("""
Chạy benchmark:
  # OCR only (nếu chỉ test PARSeq):
  python example_run.py --mode ocr --crops test_data/crops/

  # Full pipeline:
  python example_run.py --mode full --data test_data/
""")