import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Squeeze and Excitation Block - Được sửa để đảm bảo kích thước đúng
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Đảm bảo reduction không làm quá nhỏ số kênh
        reduction_dim = max(channel // reduction, 4)
        
        # Sử dụng Conv2d thay vì Linear để tránh lỗi reshape
        self.fc = nn.Sequential(
            nn.Conv2d(channel, reduction_dim, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_dim, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

# MobileBlock (Depthwise Separable Convolution)
class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobileBlock, self).__init__()
        
        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                      padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

# Lightweight Emotion Recognition Model (dưới 3M tham số)
class LightweightEmotionNet(nn.Module):
    def __init__(self, num_classes=7, input_channels=1, width_mult=0.5):
        super(LightweightEmotionNet, self).__init__()
        
        # Đảm bảo width_mult không quá lớn
        self.width_mult = min(width_mult, 0.75)
        
        # Kích thước cơ bản của các kênh
        def make_divisible(x):
            return int(max(8, int(x * self.width_mult + 0.5) // 8 * 8))
        
        # Định nghĩa số kênh
        self.channels = [
            make_divisible(32),   # Stem
            make_divisible(64),   # Block 1
            make_divisible(128),  # Block 2
            make_divisible(256),  # Block 3
            make_divisible(512),  # Block 4
        ]
        
        # Stem layer - giảm kích thước nhưng tăng số kênh
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, self.channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.channels[0]),
            nn.ReLU6(inplace=True)
        )
        
        # Xây dựng các khối MobileNet
        self.block1 = nn.Sequential(
            MobileBlock(self.channels[0], self.channels[1], stride=1),
            SEBlock(self.channels[1], reduction=8)
        )
        
        self.block2 = nn.Sequential(
            MobileBlock(self.channels[1], self.channels[2], stride=2),
            MobileBlock(self.channels[2], self.channels[2], stride=1),
            SEBlock(self.channels[2], reduction=8)
        )
        
        self.block3 = nn.Sequential(
            MobileBlock(self.channels[2], self.channels[3], stride=2),
            MobileBlock(self.channels[3], self.channels[3], stride=1),
            SEBlock(self.channels[3], reduction=16)
        )
        
        self.block4 = nn.Sequential(
            MobileBlock(self.channels[3], self.channels[4], stride=2),
            MobileBlock(self.channels[4], self.channels[4], stride=1),
            SEBlock(self.channels[4], reduction=16)
        )
        
        # Global pooling và classifier
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        
        # Sử dụng bottleneck để giảm tham số
        self.classifier = nn.Sequential(
            nn.Linear(self.channels[4], num_classes)
        )
        
        # Khởi tạo trọng số
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
    
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

# Lightweight model with attention mechanism for facial emotion recognition
class AttentionEmotionNet(nn.Module):
    def __init__(self, num_classes=7, input_channels=1, width_mult=0.5):
        super(AttentionEmotionNet, self).__init__()
        
        # Base model
        self.base = LightweightEmotionNet(num_classes, input_channels, width_mult)
        
        # Channels 
        base_channels = self.base.channels
        
        # Spatial attention module (after block2)
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(base_channels[2], 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # Multi-head attention (simplified version)
        self.multihead_attention = nn.Sequential(
            nn.Conv2d(base_channels[3], base_channels[3], kernel_size=1, bias=False),
            nn.BatchNorm2d(base_channels[3]),
            nn.ReLU6(inplace=True),
            SEBlock(base_channels[3], reduction=8)
        )
        
        # Fusion layer (kết hợp đặc trưng từ nhiều levels)
        reduced_dim = base_channels[4] // 4
        self.fusion = nn.Sequential(
            nn.Linear(base_channels[4], reduced_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(reduced_dim, num_classes)
        )
    
    def forward(self, x):
        # Stem và block1
        x = self.base.stem(x)
        feat1 = self.base.block1(x)
        
        # Block2 với spatial attention
        feat2 = self.base.block2(feat1)
        spatial_attn = self.spatial_attention(feat2)
        feat2_refined = feat2 * spatial_attn
        
        # Block3 với multi-head attention
        feat3 = self.base.block3(feat2_refined)
        feat3_refined = self.multihead_attention(feat3)
        
        # Block4 (high-level features)
        feat4 = self.base.block4(feat3_refined)
        
        # Global pooling và classification
        x = self.base.global_pool(feat4)
        x = x.view(x.size(0), -1)
        x = self.base.dropout(x)
        x = self.fusion(x)
        
        return x

# Kiểm tra mô hình và số tham số
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cấu hình mô hình
MODEL_TYPE = 'attention'  # 'basic' hoặc 'attention'
WIDTH_MULT = 0.5  # Giảm xuống để có ít tham số hơn

# Khởi tạo mô hình theo lựa chọn
if MODEL_TYPE == 'basic':
    model = LightweightEmotionNet(num_classes=7, input_channels=1, width_mult=WIDTH_MULT).to(device)
else:  # 'attention'
    model = AttentionEmotionNet(num_classes=7, input_channels=1, width_mult=WIDTH_MULT).to(device)

# In số tham số
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Under 3M threshold: {'Yes' if total_params < 3000000 else 'No'}")

# Kiểm tra với tensor ngẫu nhiên
dummy_input = torch.randn(1, 1, 48, 48).to(device)
with torch.no_grad():
    model.eval()
    output = model(dummy_input)

print(f"Output shape: {output.shape}")
print(f"Model type: {MODEL_TYPE}")
print(f"Width multiplier: {WIDTH_MULT}")
