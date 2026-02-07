import torch
import torch.nn as nn
import torchvision
import os
import numpy as np
from PIL import Image
import time

# --- CẤU HÌNH ---
IMAGE_PATH = "test_image.jpg"   # <-- Tên ảnh đầu vào
MODEL_PATH = "Zero-DCE_extension-main/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth"

print(f"\n--- ZERO-DCE++ FINAL (SAVE TO FOLDER) ---")

# ==========================================
# 1. MODEL CHUẨN
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
        self.e_conv7 = C_DCE_Sep_Conv(number_f * 2, 3) 

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        
        # Dùng Sigmoid để ảnh sáng đẹp, chuẩn [0, 1]
        x_r = torch.sigmoid(self.e_conv7(torch.cat([x1, x6], 1)))
        return x_r

# ==========================================
# 2. HÀM CHẠY
# ==========================================
def run_one_image(image_path):
    if not os.path.exists(image_path):
        print(f"❌ LỖI: Không tìm thấy ảnh tại {image_path}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Đang chạy trên: {device}")

    # Load ảnh
    img_org = Image.open(image_path).convert('RGB')
    data_lowlight = (np.asarray(img_org) / 255.0)
    data_lowlight = torch.from_numpy(data_lowlight).float()

    # Resize chẵn
    scale_factor = 12
    h = (data_lowlight.shape[0] // scale_factor) * scale_factor
    w = (data_lowlight.shape[1] // scale_factor) * scale_factor
    data_lowlight = data_lowlight[0:h, 0:w, :]
    
    data_lowlight = data_lowlight.permute(2, 0, 1)
    data_lowlight = data_lowlight.to(device).unsqueeze(0)

    # Load Model
    model = enhance_net_nopool().to(device)
    
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict, strict=True)
        model.eval()
        print("✅ Load weights OK!")
    except Exception as e:
        print(f"❌ Lỗi weights: {e}")
        return

    # Inference
    start = time.time()
    with torch.no_grad():
        enhanced_image = model(data_lowlight)
    end_time = time.time() - start
    print(f"⏱️ Xử lý xong: {end_time:.4f}s")

    # --- SAVE IMAGE VÀO FOLDER TEST_DCE ---
    output_folder = "test_dce"
    os.makedirs(output_folder, exist_ok=True) # Tạo folder nếu chưa có

    filename = "result_" + os.path.basename(image_path)
    result_path = os.path.join(output_folder, filename)

    torchvision.utils.save_image(enhanced_image, result_path)
    print(f"🎉 Đã lưu ảnh vào folder '{output_folder}' tại: {result_path}")

if __name__ == '__main__':
    run_one_image(IMAGE_PATH)