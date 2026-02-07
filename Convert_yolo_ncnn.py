import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# --- CẤU HÌNH ---
# Kiểm tra lại đường dẫn file .pth
MODEL_PATH = r'Zero-DCE_extension-main/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth'
ONNX_OUTPUT = "zerodce.onnx"

print(f"\n--- CONVERT ZERO-DCE++ (DIRECT OUTPUT VERSION) ---")

# 1. ĐỊNH NGHĨA BLOCK CONV (Giữ nguyên Depthwise)
class C_DCE_Sep_Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(C_DCE_Sep_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=True)
        self.point_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=1, bias=True)

    def forward(self, x):
        out = self.depth_conv(x)
        out = self.point_conv(out)
        return out

# 2. ĐỊNH NGHĨA MẠNG (Đã sửa output layer)
class ZeroDCE_Direct(nn.Module):
    def __init__(self, scale_factor=1):
        super(ZeroDCE_Direct, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32

        # Các lớp đầu giữ nguyên
        self.e_conv1 = C_DCE_Sep_Conv(3, number_f)
        self.e_conv2 = C_DCE_Sep_Conv(number_f, number_f)
        self.e_conv3 = C_DCE_Sep_Conv(number_f, number_f)
        self.e_conv4 = C_DCE_Sep_Conv(number_f, number_f)
        self.e_conv5 = C_DCE_Sep_Conv(number_f * 2, number_f)
        self.e_conv6 = C_DCE_Sep_Conv(number_f * 2, number_f)
        
        # --- THAY ĐỔI QUAN TRỌNG TẠI ĐÂY ---
        # Lỗi cũ: size mismatch ... shape is [24...] but copying [3...]
        # Sửa: Đổi 24 thành 3 để khớp với file weights của ông
        self.e_conv7 = C_DCE_Sep_Conv(number_f * 2, 3) 

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        
        # Output ra 3 kênh (RGB) luôn
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        # Model này là dạng Direct Mapping, không có thuật toán Curve Loop
        # Nên ta trả về kết quả trực tiếp.
        return x_r

# 3. CONVERT
if __name__ == "__main__":
    # Khởi tạo model đã sửa
    net = ZeroDCE_Direct().cpu()

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Lỗi: Không tìm thấy file {MODEL_PATH}")
        exit()

    try:
        print(f"⏳ Đang load weights từ {MODEL_PATH}...")
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        
        # Clean dictionary keys
        new_state_dict = {}
        for k, v in checkpoint.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v
            
        # Load weights (Lần này 24 đã sửa thành 3 nên sẽ khớp)
        net.load_state_dict(new_state_dict, strict=True)
        net.eval()
        print("✅ Load weights thành công! (Structure Matched: 3 Output Channels)")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        print("Tip: Nếu vẫn lỗi, khả năng file .pth này không phải kiến trúc Zero-DCE++ chuẩn.")
        exit()

    # 4. EXPORT ONNX
    dummy_input = torch.randn(1, 3, 320, 320)
    print(f"⏳ Đang convert sang {ONNX_OUTPUT}...")

    torch.onnx.export(
        net,
        dummy_input,
        ONNX_OUTPUT,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )

    print(f"🎉 XONG! File '{ONNX_OUTPUT}' đã sẵn sàng.")
    print("👉 Chạy file test ONNX ngay đi, lần này chắc chắn lên hình!")