import os
from ultralytics import YOLO

# --- ĐƯỜNG DẪN CÁC FOLDER NCNN CỦA ÔNG ---
# Ông kiểm tra lại xem tên folder có đúng như này không nhé
MODELS_CFG = {
    'YOLOv8_NCNN':  r'runs/detect/yolov8_trained/weights/best_ncnn_model',
    'YOLOv11_NCNN': r'runs/detect/yolov11_trained/weights/best_ncnn_model',
    'YOLOv26_NCNN': r'runs/detect/yolov26_trained/weights/best_ncnn_model'
}

DATA_YAML = 'data_train/data.yaml'

def benchmark_ncnn_only():
    results = []
    print(f"\n>>> Đang quét các folder NCNN đã export...")

    for name, path in MODELS_CFG.items():
        if not os.path.exists(path):
            print(f"(!) Bỏ qua {name}: Không tìm thấy folder tại {path}")
            continue
            
        try:
            print(f"--- Đang đo {name} ---")
            # 1. Load trực tiếp từ folder NCNN
            model = YOLO(path)
            
            # 2. Chạy Validate để lấy thông số Speed
            # imgsz=640 là chuẩn, device='cpu' để tránh lỗi Device ID
            metrics = model.val(data=DATA_YAML, imgsz=640, device='cpu', plots=False, verbose=False)
            
            results.append({
                'Model': name,
                'Inference (ms)': metrics.speed['inference'],
                'Pre-process (ms)': metrics.speed['preprocess'],
                'Post-process (ms)': metrics.speed['postprocess']
            })
        except Exception as e:
            print(f"(!) Lỗi khi đo {name}: {e}")

    # --- IN BẢNG TỔNG KẾT ---
    print("\n" + "="*75)
    print(f"| {'MODEL NCNN':<15} | {'Inference (ms)':<15} | {'Pre-process':<15} | {'Post-process':<12} |")
    print("-" * 75)
    for r in results:
        print(f"| {r['Model']:<15} | {r['Inference (ms)']:<15.2f} | {r['Pre-process (ms)']:<15.2f} | {r['Post-process (ms)']:<12.2f} |")
    print("="*75)

if __name__ == "__main__":
    benchmark_ncnn_only()