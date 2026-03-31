# 1. Thông tin Model (Model Specs)
Định dạng: ONNX.

Input Name: Phụ thuộc vào file xuất ra (thường là images hoặc có thể lấy tự động qua session.get_inputs()[0].name).

Input Shape: (Batch_Size, 3, 32, 128) (N, C, H, W).

Output Shape: (Batch_Size, Sequence_Length, Num_Classes).

# 2. Bảng Ký Tự (Charset)
Mô hình này không sử dụng bảng mã ASCII mặc định mà dùng một chuỗi CHARSET được định nghĩa riêng. Bạn bắt buộc phải dùng chuỗi này để map index đầu ra thành ký tự chữ/số.

```Python
CHARSET = r"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
```

Lưu ý quan trọng: Index 0 của model trả về được quy ước là ký tự kết thúc chuỗi (EOS - End of Sequence).

-> Các index từ 1 trở đi sẽ tương ứng với CHARSET[index - 1].

# 3. Quá trình Tiền xử lý (Preprocessing)
Ảnh biển số sau khi được cắt ra từ camera cần trải qua các bước sau trước khi đẩy vào model:

## 3.1. Chuẩn hóa ảnh (Normalization) - file utils.py
Mọi ảnh (dù là biển 1 dòng hay 2 dòng) đều phải được đưa về kích thước chuẩn và chuẩn hóa giá trị pixel bằng thuật toán blobFromImage:

- Kích thước mục tiêu (Target Size): Rộng 128, Cao 32 (128x32).

- Scale Factor: 1.0 / 127.5.

- Mean Subtraction: Trừ đi giá trị trung bình (127.5, 127.5, 127.5).

- SwapRB: True (Đổi kênh màu từ BGR của OpenCV sang RGB).

## 3.2. Xử lý logic Biển dài (1 dòng) vs Biển vuông (2 dòng) - file pipeline.py
Dựa vào tỷ lệ kích thước (Aspect Ratio = Rộng / Cao), hệ thống chia làm 2 trường hợp xử lý:

- Trường hợp 1 (Biển 1 dòng): Nếu Tỷ lệ > 2.2 -> Đưa trực tiếp toàn bộ ảnh biển số qua hàm chuẩn hóa và đẩy vào model.

- Trường hợp 2 (Biển 2 dòng): Nếu Tỷ lệ <= 2.2 -> Cắt ảnh gốc thành 2 nửa:

+ Dòng trên (Top): Từ trên cùng đến 60% chiều cao ảnh (0 : int(h * 0.60)).

+ Dòng dưới (Bottom): Từ mốc 40% chiều cao ảnh đến dưới cùng (int(h * 0.40) :).

- Lý do vùng cắt đè lên nhau (từ 40% đến 60%) là để chống mất nét các ký tự nằm giữa biển.

- Sau khi cắt, chuẩn hóa cả 2 ảnh nhỏ này thành 2 tensor 128x32 và gom thành 1 Batch (N=2) để đẩy qua GPU/CPU trong 1 lần chạy duy nhất.

# 4. Quá trình Hậu xử lý và Giải mã (Postprocessing & Decoding) - file inference.py
Đầu ra của mô hình là một ma trận logits chứa xác suất của các ký tự. Hàm giải mã (decode_parseq) hoạt động như sau:

1) Lặp qua từng bước thời gian (Sequence Length):

Tại mỗi bước, tìm vị trí có giá trị logit lớn nhất (max_idx = argmax(logits_step)).

2) Kiểm tra điều kiện dừng:

Nếu max_idx == 0, dừng việc đọc ngay lập tức vì đã gặp ký tự kết thúc (EOS).

3) Map ký tự:

Nếu max_idx > 0, nối thêm ký tự CHARSET[max_idx - 1] vào chuỗi kết quả.

4) Tính điểm tự tin (Confidence Score):

- Tính giá trị Softmax tại vị trí max_idx để lấy xác suất của ký tự đó: char_conf = 1.0 / sum(exp(logits_step - max_val)).

- Lưu danh sách char_conf của toàn bộ các ký tự hợp lệ.

- Điểm tự tin của toàn bộ biển số được tính bằng công thức thiên vị ký tự yếu nhất: (Điểm trung bình * 0.4) + (Điểm thấp nhất * 0.6).

# 5. Làm sạch kết quả (Text Cleaning)
Sau khi giải mã OCR, chuỗi text thô cần được chuẩn hóa định dạng:

- Xóa ký tự rác: Chỉ giữ lại các ký tự là chữ cái và chữ số (isalnum()), và in hoa toàn bộ.

- Định dạng biển 2 dòng:

+ Dòng trên: Chỉ lấy 4 ký tự đầu, và chèn dấu - vào giữa (ví dụ: 51F-1).

+ Dòng dưới: Chỉ lấy các ký tự số, nếu có từ 5 số trở lên, tự động chèn dấu chấm (ví dụ: 123.45).

+ Kết hợp lại thành Dòng_Trên + Khoảng_Trắng + Dòng_Dưới (VD: 51F-1 123.45).

# 6. Đặc tả chi tiết Input / Output (Tensor Specs)
Để setup bộ đệm (buffer) trong ONNX Runtime, người chạy model cần nắm chính xác các thông số sau:

INPUT TENSOR:

- Data Type: float32

- Shape: [N, 3, 32, 128] (Trong đó N là Batch Size. N=1 cho biển 1 dòng, N=2 nếu ghép dòng trên và dòng dưới của biển 2 dòng vào chạy cùng lúc).

- Giá trị (Values): Các pixel sau khi đưa qua hàm blobFromImage với scale 1.0 / 127.5 và mean 127.5 sẽ nằm trong khoảng giá trị [-1.0, 1.0].

OUTPUT TENSOR:

- Data Type: float32

- Shape: [N, Sequence_Length, Num_Classes]

+ Sequence_Length: Chiều dài tối đa của chuỗi (ví dụ: 26).

+ Num_Classes: Số lượng ký tự trong CHARSET + 1 (ký tự EOS ở vị trí index 0).

- Giá trị (Values): Đầu ra là Raw Logits (chưa qua Softmax). Bạn cần dùng Argmax trên chiều Num_Classes để lấy ra index của ký tự tại mỗi bước.