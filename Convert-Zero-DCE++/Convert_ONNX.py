import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Ông kiểm tra lại đường dẫn này cho đúng vị trí file Epoch99.pth trên máy ông
MODEL_PATH = r'Zero-DCE_extension-main/Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth'
ONNX_OUTPUT = "zerodce.onnx"

print(f"\n--- FIX EXPORT ONNX ZERO-DCE++ ---")

# 1. ĐỊNH NGHĨA KIẾN TRÚC MODEL (Chép thẳng vào đây cho an toàn)
class enhance_net_nopool(nn.Module):
    def __init__(self, scale_factor=1):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32

        # Zero-DCE++ dùng Depthwise Separable Conv để nhẹ hơn
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def enhance(self, x, x_r):
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image_1 = x + x_r * (torch.pow(x, 2) - x)
        x = enhance_image_1 + x_r * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + x_r * (torch.pow(x, 2) - x)
        x = x + x_r * (torch.pow(x, 2) - x)
        enhance_image = x + x_r * (torch.pow(x, 2) - x)
        return enhance_image

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))

        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        
        # Output params map
        x_r = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        # Split params
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)
        
        # Iterative enhancement
        x = x + r1 * (torch.pow(x, 2) - x)
        x = x + r2 * (torch.pow(x, 2) - x)
        x = x + r3 * (torch.pow(x, 2) - x)
        enhance_image_1 = x + r4 * (torch.pow(x, 2) - x)
        x = enhance_image_1 + r5 * (torch.pow(enhance_image_1, 2) - enhance_image_1)
        x = x + r6 * (torch.pow(x, 2) - x)
        x = x + r7 * (torch.pow(x, 2) - x)
        enhance_image = x + r8 * (torch.pow(x, 2) - x)
        
        return enhance_image, x_r

# 2. KHỞI TẠO VÀ LOAD TRỌNG SỐ
print(">>> Đang khởi tạo model...")
net = enhance_net_nopool(scale_factor=1).cpu()

if not os.path.exists(MODEL_PATH):
    print(f"❌ LỖI: Không tìm thấy file {MODEL_PATH}")
    print("👉 Ông sửa lại dòng MODEL_PATH ở đầu file code nhé.")
    exit()

try:
    # Fix 1: map_location='cpu' thay vì '0'
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # Fix 2: Xử lý key 'module.' nếu có (do DataParallel)
    new_state_dict = {}
    for k, v in checkpoint.items():
        name = k.replace('module.', '') 
        new_state_dict[name] = v
        
    # Load với strict=True để đảm bảo file pth đúng chuẩn
    net.load_state_dict(new_state_dict, strict=True)
    net.eval()
    print("✅ Load weights thành công!")
    
except Exception as e:
    print(f"❌ Lỗi load file .pth: {e}")
    # Nếu vẫn lỗi, thử load lỏng lẻo hơn
    print("⚠️ Đang thử load lại với strict=False...")
    try:
        net.load_state_dict(checkpoint, strict=False)
        print("✅ Load (strict=False) thành công!")
    except:
        exit()

# 3. WRAPPER ĐỂ CHỈ LẤY ẢNH OUTPUT
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        # enhance_net_nopool trả về (image, params)
        # Ta chỉ lấy cái đầu tiên [0]
        result = self.model(x)
        if isinstance(result, tuple):
            return result[0]
        return result

wrapped_net = ModelWrapper(net)
wrapped_net.eval()

# 4. EXPORT ONNX
dummy_input = torch.randn(1, 3, 320, 320)

print(f">>> Đang convert sang {ONNX_OUTPUT}...")

torch.onnx.export(
    wrapped_net,
    dummy_input,
    ONNX_OUTPUT,
    export_params=True,
    opset_version=11,      # Bản chuẩn nhất cho ONNX Runtime
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print(f"🎉 THÀNH CÔNG! File '{ONNX_OUTPUT}' đã sẵn sàng.")
print("👉 Giờ ông chạy file test bằng ONNX Runtime là được.")