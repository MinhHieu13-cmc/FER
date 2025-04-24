import os
import sys

# Thêm thư mục gốc vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Kiểm tra import
from models.backbone.fer_mobile_vit import FERMobileViTDAN
import torch

# Khởi tạo model
model = FERMobileViTDAN(num_classes=3)
print(f"Đã khởi tạo thành công {model.__class__.__name__}")

# Test forward pass
dummy_input = torch.randn(1, 3, 160, 160)
with torch.no_grad():
    output = model(dummy_input)
print(f"Kích thước đầu ra: {output.shape}")