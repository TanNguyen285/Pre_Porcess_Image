import cv2
import numpy as np
import onnxruntime as ort
import os

# --- CẤU HÌNH ---
# Ông nhớ đổi đường dẫn trỏ đúng vào file .onnx nhé
ONNX_PATH  = r"Convert-Zero-DCE++\zerodce.onnx" 
IMAGE_PATH = "test_image.jpg"

print("\n--- ZERO-DCE++ ONNX RUNTIME TEST ---")

# 1. Kiểm tra file
if not os.path.exists(ONNX_PATH):
    print(f"❌ LỖI: Không tìm thấy file model tại: {ONNX_PATH}")
    print("👉 Hãy chắc chắn ông đã có file .onnx (nếu chưa có thì export từ .pth sang)")
    exit()

# 2. Load Model ONNX
# Tự động chọn GPU (CUDA) nếu có, không thì chạy CPU
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
try:
    session = ort.InferenceSession(ONNX_PATH, providers=providers)
except Exception as e:
    print(f"⚠️ Lỗi khởi tạo (có thể do chưa cài CUDA), chuyển sang CPU...")
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])

# Lấy tên Input/Output tự động (Khỏi lo sai tên layer)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"✅ Model Loaded! Input: '{input_name}' -> Output: '{output_name}'")

# 3. Đọc ảnh
img = cv2.imread(IMAGE_PATH)
if img is None:
    print("❌ LỖI: Không tìm thấy ảnh input!")
    exit()

h_orig, w_orig = img.shape[:2]

# 4. Chuẩn bị Input (Pre-processing)
# Resize về 320x320 (Kích thước chuẩn của Zero-DCE)
target_w, target_h = 320, 320
img_resized = cv2.resize(img, (target_w, target_h))

# Đổi BGR -> RGB
img_in = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# Normalize: Chia 255 để về khoảng [0, 1]
img_in = img_in.astype(np.float32) / 255.0

# Transpose: Đổi trục từ (H, W, C) -> (C, H, W) 
# (Đây là bước NCNN tự làm, nhưng ONNX phải làm thủ công)
img_in = img_in.transpose(2, 0, 1)

# Thêm dimension Batch: (3, 320, 320) -> (1, 3, 320, 320)
img_in = np.expand_dims(img_in, axis=0)

# 5. Chạy Model (Inference)
# Trả về list kết quả, lấy phần tử đầu tiên [0]
outputs = session.run([output_name], {input_name: img_in})
output_tensor = outputs[0]

# 6. Xử lý Output (Post-processing)
# Bỏ dimension Batch: (1, 3, 320, 320) -> (3, 320, 320)
result = np.squeeze(output_tensor)

# Đổi trục ngược lại: (C, H, W) -> (H, W, C) để hiển thị
result = result.transpose(1, 2, 0)

# Nhân 255 và clip giá trị để không bị lỗi màu
result = (result * 255.0).clip(0, 255).astype(np.uint8)

# Đổi RGB -> BGR
result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

# Resize về kích thước gốc
result_final = cv2.resize(result_bgr, (w_orig, h_orig))

# 7. Hiển thị
cv2.putText(img, "ORIGINAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(result_final, "ONNX ENHANCED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

combined = np.hstack((img, result_final))
cv2.imshow("Zero-DCE ONNX Result", combined)

print("👉 Đã hiện ảnh. Bấm phím bất kỳ để thoát.")
cv2.waitKey(0)
cv2.destroyAllWindows()