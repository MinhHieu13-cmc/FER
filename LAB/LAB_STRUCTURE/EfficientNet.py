import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0

class OptimizedEmotionNet(nn.Module):
    def __init__(self, num_classes=8):
        super(OptimizedEmotionNet, self).__init__()
        # Sử dụng EfficientNet-B0 nhưng không tải pretrained weights để tránh lỗi khi tải
        base = efficientnet_b0(weights=None)
        
        # Lấy các lớp đầu tiên của EfficientNet (các khối đầu tiên)
        # Kiểm tra cấu trúc đầu ra của từng khối
        self.features = nn.Sequential(
            base.features[0],  # Conv stem
            base.features[1],  # Block 1 - output: 16 channels
            base.features[2],  # Block 2 - output: 24 channels  
            base.features[3],  # Block 3 - output: 40 channels
            base.features[4],  # Block 4 - output: 80 channels
            base.features[5],  # Block 5 - output: 112 channels
        )
        
        # Số kênh đầu ra từ block 5 là 112
        num_channels = 112
        
        # Kết hợp các tính năng
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Sửa lại channel_attention để phù hợp với 112 channels
        self.channel_attention = ChannelAttention(num_channels)
        
        # Feature fusion
        mid_channels = 64
        self.fusion = nn.Sequential(
            nn.Conv2d(num_channels, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),  # Sử dụng ReLU thay vì SiLU để tương thích tốt hơn
            nn.Dropout(0.25)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(mid_channels, num_classes)
        )
        
        # Khởi tạo trọng số
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.features(x)
        
        # Apply attention
        x = self.channel_attention(x) * x
        
        x = self.fusion(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Squeeze-and-Excitation Channel Attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Đảm bảo reduction_ratio không làm giảm quá nhiều kênh
        reduction_channels = max(in_channels // reduction_ratio, 8)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduction_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

# Thiết lập mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OptimizedEmotionNet(num_classes=8).to(device)

# Chuyển sang chế độ eval để kiểm tra
model.eval()

# Test với input giả lập
dummy_input = torch.randn(1, 3, 112, 112).to(device)
with torch.no_grad():
    output = model(dummy_input)

# In số tham số & output shape
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Output shape: {output.shape}")
print(f"Model size under 3M threshold: {'Yes' if total_params < 3000000 else 'No'}")

# So sánh với model gốc
original_model = efficientnet_b0(weights=None)
original_model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(1280, 8)
)
original_model = original_model.to(device)
original_model.eval()

with torch.no_grad():
    original_output = original_model(dummy_input)

original_params = sum(p.numel() for p in original_model.parameters())
print(f"Original model parameters: {original_params:,}")
print(f"Parameter reduction: {100 * (1 - total_params / original_params):.2f}%")
