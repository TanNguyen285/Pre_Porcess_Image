import torch
import torchvision
import os
import model # Model Zero-DCE++ của bạn
import numpy as np
from PIL import Image
import glob
import time
from ultralytics import YOLO # Thư viện YOLOv8/v9/v10/v11

def process_image(image_path, output_root, DCE_net, yolo_model):
    # --- 1. TIỀN XỬ LÝ ---
    data_lowlight = Image.open(image_path).convert('RGB')
    scale_factor = 12
    img_numpy = np.asarray(data_lowlight) / 255.0
    input_tensor = torch.from_numpy(img_numpy).float()

    h = (input_tensor.shape[0] // scale_factor) * scale_factor
    w = (input_tensor.shape[1] // scale_factor) * scale_factor
    input_tensor = input_tensor[0:h, 0:w, :].permute(2, 0, 1).cuda().unsqueeze(0)

    # --- 2. CHẠY ZERO-DCE++ (LÀM SÁNG) ---
    torch.cuda.synchronize()
    start_dce = time.time()
    
    enhanced_image, _ = DCE_net(input_tensor)
    
    torch.cuda.synchronize()
    dce_time = (time.time() - start_dce) * 1000 # Chuyển sang ms

    # --- 3. CHẠY YOLO (NHẬN DIỆN) ---
    # Chuyển tensor kết quả sang định dạng YOLO có thể đọc (PIL Image hoặc Numpy)
    # Lưu ý: enhanced_image đang là Tensor [1, 3, H, W]
    enhanced_pil = torchvision.transforms.ToPILImage()(enhanced_image.squeeze(0).cpu())
    
    torch.cuda.synchronize()
    start_yolo = time.time()
    
    # Chạy YOLO trên ảnh đã được làm sáng
    yolo_results = yolo_model(enhanced_pil, verbose=False)
    
    torch.cuda.synchronize()
    yolo_time = (time.time() - start_yolo) * 1000 # Chuyển sang ms

    # --- 4. LƯU KẾT QUẢ ---
    file_name = os.path.basename(image_path)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    
    # Vẽ bounding box của YOLO lên ảnh và lưu
    # Nếu muốn lưu ảnh sạch (chỉ làm sáng), dùng: torchvision.utils.save_image(enhanced_image, result_path)
    res_plotted = yolo_results[0].plot() # Ảnh đã có khung nhận diện
    cv2_img = Image.fromarray(res_plotted[:, :, ::-1]) # Chuyển BGR sang RGB
    cv2_img.save(os.path.join(output_root, file_name))

    return dce_time, yolo_time

if __name__ == '__main__':
    # Khởi tạo thư mục và Model
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    input_path = 'Zero-DCE_extension-main/Zero-DCE++/data/test_data/'
    output_external = 'C:/Users/LagCT/Desktop/DCE_YOLO_Results'
    
    with torch.no_grad():
        # Load Zero-DCE++
        DCE_net = model.enhance_net_nopool(12).cuda()
        DCE_net.load_state_dict(torch.load('Zero-DCE_extension-main/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth', weights_only=True))
        DCE_net.eval()

        # Load YOLO (Sẽ tự tải weights 'yolov8n.pt' nếu chưa có)
        yolo_model = YOLO('C:\\Users\\LagCT\\Desktop\\Image Pre-processing\\runs\\detect\\yolov26_trained\\weights\\best.pt').to('cuda') 

        test_list = glob.glob(os.path.join(input_path, "**/*.*"), recursive=True)
        test_list = [f for f in test_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"{'File Name':<25} | {'DCE++ (ms)':<12} | {'YOLO (ms)':<12}")
        print("-" * 55)

        total_dce, total_yolo = 0, 0
        for image in test_list:
            d_time, y_time = process_image(image, output_external, DCE_net, yolo_model)
            
            total_dce += d_time
            total_yolo += y_time
            
            name = os.path.basename(image)
            print(f"{name[:24]:<25} | {d_time:>10.2f} | {y_time:>10.2f}")

        # Tổng kết
        n = len(test_list)
        if n > 0:
            print("-" * 55)
            print(f"{'AVERAGE':<25} | {total_dce/n:>10.2f} | {total_yolo/n:>10.2f} ms/ảnh")
            print(f"Tổng cộng xử lý 1 ảnh: {(total_dce + total_yolo)/n:.2f} ms")