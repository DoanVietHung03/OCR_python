import onnx

def make_dynamic_batch(input_path, output_path):
    print(f"Đang tải model từ: {input_path}")
    model = onnx.load(input_path)

    # 1. Đổi input batch size thành dạng chuỗi (dynamic)
    for input_node in model.graph.input:
        dim = input_node.type.tensor_type.shape.dim
        if len(dim) > 0:
            # Gán dim[0] (thường là batch_size) thành một chuỗi đại diện cho dynamic
            dim[0].dim_param = 'dynamic_batch'

    # 2. Đổi output batch size thành dạng chuỗi (dynamic)
    for output_node in model.graph.output:
        dim = output_node.type.tensor_type.shape.dim
        if len(dim) > 0:
            dim[0].dim_param = 'dynamic_batch'

    # Lưu lại file mới
    onnx.save(model, output_path)
    print(f"Đã lưu thành công model Dynamic Batch tại: {output_path}")

if __name__ == "__main__":
    # Điền đường dẫn file cũ và file mới vào đây
    old_model = "weights/yolov9_detect_plate.onnx"
    new_model = "weights/yolov9_detect_plate_dynamic.onnx"
    
    make_dynamic_batch(old_model, new_model)