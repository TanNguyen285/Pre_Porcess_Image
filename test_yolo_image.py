import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import numpy as np
from PIL import Image
import time

# --- CẤU HÌNH ---
IMAGE_PATH = "test_image.jpg"  # <-- Điền tên ảnh của ông vào đây
MODEL_PATH = "Zero-DCE_extension-main/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth" # <-- Đường dẫn file weights

# ==========================================
# 1. ĐỊNH NGHĨA MODEL (PHẢI ĐỂ Ở ĐÂY ĐỂ FIX LỖI 3 KÊNH)
# (Nếu import model.py gốc sẽ bị lỗi size mismatch 24 vs 3)
# ==========================================
class C_DCE_Sep_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C_DCE_Sep_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=True)
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=1, bias=True)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out

class enhance_net_nopool(nn.Module):
    def __init__(self, scale_factor=1):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = C_DCE_Sep_Conv(3, number_f)
        self.e_conv2 = C_DCE_Sep_Conv(number_f, number_f)
        self.e_conv3 = C_DCE_Sep_Conv(number_f, number_f)
        self.e_conv4 = C_DCE_Sep_Conv(number_f, number_f)
        self.e_conv5 = C_DCE_Sep_Conv(number_f * 2, number_f)
        self.e_conv6 = C_DCE_Sep_Conv(number_f * 2, number_f)
        # QUAN TRỌNG: Output = 3 (RGB) để khớp file Epoch99.pth của ông
        self.e_conv7 = C_DCE_Sep_Conv(number_f * 2, 3) 

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        # Output ảnh trực tiếp
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return x_r

# ==========================================
# 2. CODE LOGIC GỐC CỦA ÔNG (Đã sửa thành 1 ảnh)
# ==========================================
def lowlight(image_path):
    # Tự động nhận diện thiết bị (để tránh lỗi nếu máy ko có GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Đang chạy trên: {device}")
    
    scale_factor = 12
    
    # --- Pre-processing (Giữ nguyên logic gốc) ---
    data_lowlight = Image.open(image_path).convert('RGB') # Fix: convert RGB tránh lỗi ảnh PNG 4 kênh
    data_lowlight = (np.asarray(data_lowlight)/255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()

    h=(data_lowlight.shape[0]//scale_factor)*scale_factor
    w=(data_lowlight.shape[1]//scale_factor)*scale_factor
    data_lowlight = data_lowlight[0:h,0:w,:]
    data_lowlight = data_lowlight.permute(2,0,1)
    
    # Đẩy vào device (GPU/CPU)
    data_lowlight = data_lowlight.to(device).unsqueeze(0)

    # --- Load Model ---
    # Thay vì import model, ta dùng class định nghĩa ở trên
    DCE_net = enhance_net_nopool(scale_factor).to(device)
    
    print(f"⏳ Đang load weights từ {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Fix lỗi key 'module.'
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
        
    DCE_net.load_state_dict(new_state_dict, strict=True)
    DCE_net.eval()

    # --- Inference ---
    start = time.time()
    # Logic gốc trả về 2 biến, nhưng model của ông chỉ trả về 1 ảnh (vì là bản 3 kênh)
    enhanced_image = DCE_net(data_lowlight) 
    end_time = (time.time() - start)
    print(f"⏱️ Xử lý xong trong: {end_time:.4f} giây")

    # --- Save Image ---
    result_path = "result_" + os.path.basename(image_path)
    torchvision.utils.save_image(enhanced_image, result_path)
    print(f"🎉 Đã lưu ảnh tại: {result_path}")

if __name__ == '__main__':
    if os.path.exists(IMAGE_PATH):
        with torch.no_grad():
            lowlight(IMAGE_PATH)
    else:
        print(f"❌ Không tìm thấy file ảnh: {IMAGE_PATH}")