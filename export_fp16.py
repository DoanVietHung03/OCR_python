import onnx
from onnxconverter_common import float16
import os

def convert_to_fp16(input_path, output_path):
    print(f"⏳ Đang xử lý: {input_path}...")
    try:
        # 1. Tải model FP32 gốc
        model_fp32 = onnx.load(input_path)
        
        # 2. Ép kiểu toàn bộ weights/nodes sang FP16
        model_fp16 = float16.convert_float_to_float16(
            model_fp32, 
            op_block_list=['Resize', 'Cast'],
            disable_shape_infer=True
        )
        
        # 3. Lưu file mới
        onnx.save(model_fp16, output_path)
        print(f"✅ Xong! Đã lưu bản tối ưu tại: {output_path}\n")
    except Exception as e:
        print(f"❌ Lỗi khi chuyển đổi {input_path}: {e}\n")

if __name__ == "__main__":
    # Danh sách các model hiện tại trong thư mục weights của bạn
    models_to_convert = [
        ("weights/yolov9_detect_plate.onnx", "weights/yolov9_detect_plate_fp16.onnx"),
        ("weights/parseq.onnx", "weights/parseq_fp16.onnx")
    ]
    
    for in_path, out_path in models_to_convert:
        if os.path.exists(in_path):
            convert_to_fp16(in_path, out_path)
        else:
            print(f"⚠️ Không tìm thấy file: {in_path}. Vui lòng kiểm tra lại đường dẫn.")