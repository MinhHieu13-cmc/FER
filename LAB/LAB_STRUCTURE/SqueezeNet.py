import torch
import torch.nn as nn
import torch.nn.functional as F

# Fire Module với BN và Residual Connection
class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels, residual=True):
        super(FireModule, self).__init__()
        
        self.residual = residual and (in_channels == expand_channels * 2)
        
        # Squeeze layer
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, bias=False)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_channels)
        self.squeeze_activation = nn.ReLU(inplace=True)
        
        # Expand layers
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1, bias=False)
        self.expand1x1_bn = nn.BatchNorm2d(expand_channels)
        
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1, bias=False)
        self.expand3x3_bn = nn.BatchNorm2d(expand_channels)
        
        self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        
        # Squeeze
        x = self.squeeze_activation(self.squeeze_bn(self.squeeze(x)))
        
        # Expand
        x1 = self.expand1x1_bn(self.expand1x1(x))
        x3 = self.expand3x3_bn(self.expand3x3(x))
        x = torch.cat([x1, x3], dim=1)
        
        # Residual connection if possible
        if self.residual:
            x = x + identity
            
        return self.activation(x)

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Ensure reduction doesn't make channels too small
        reduction_channel = max(channel // reduction, 8)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channel, reduction_channel, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduction_channel, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y

# Enhanced SqueezeNet với Batch Normalization, SE Blocks, và Dense Connections
class EnhancedSqueezeNet(nn.Module):
    def __init__(self, num_classes=7, input_channels=1):
        super(EnhancedSqueezeNet, self).__init__()
        
        # Initial Convolution with more channels to extract better features
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Fire modules with increasing capacity
        self.fire1 = FireModule(96, 24, 64, residual=False)  # Output: 128 channels
        self.fire2 = FireModule(128, 24, 64)  # Output: 128 channels
        self.se1 = SEBlock(128)
        
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fire3 = FireModule(128, 32, 96, residual=False)  # Output: 192 channels
        self.fire4 = FireModule(192, 32, 96)  # Output: 192 channels
        self.se2 = SEBlock(192)
        
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.fire5 = FireModule(192, 48, 128, residual=False)  # Output: 256 channels
        self.fire6 = FireModule(256, 48, 128)  # Output: 256 channels
        self.se3 = SEBlock(256)
        
        self.fire7 = FireModule(256, 64, 192, residual=False)  # Output: 384 channels
        self.fire8 = FireModule(384, 64, 192)  # Output: 384 channels
        self.se4 = SEBlock(384)
        
        # Global Features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification Layer
        self.dropout = nn.Dropout(0.5)
        self.conv_final = nn.Conv2d(384, num_classes, kernel_size=1, bias=True)
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Feature extraction with attention
        x = self.fire1(x)
        x = self.fire2(x)
        x = self.se1(x)
        
        x = self.pool1(x)
        
        x = self.fire3(x)
        x = self.fire4(x)
        x = self.se2(x)
        
        x = self.pool2(x)
        
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.se3(x)
        
        x = self.fire7(x)
        x = self.fire8(x)
        x = self.se4(x)
        
        # Classification
        x = self.global_pool(x)
        x = self.dropout(x)
        x = self.conv_final(x)
        
        return x.view(x.size(0), -1)

# Feature Pyramid Enhancement (tùy chọn để tăng độ chính xác)
class FeaturePyramidSqueezeNet(nn.Module):
    def __init__(self, num_classes=7, input_channels=1):
        super(FeaturePyramidSqueezeNet, self).__init__()
        
        # Base model
        self.base = EnhancedSqueezeNet(num_classes, input_channels)
        
        # Feature fusion layers
        self.lateral_fire4 = nn.Conv2d(192, 128, kernel_size=1, bias=False)
        self.lateral_fire6 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.lateral_fire8 = nn.Conv2d(384, 128, kernel_size=1, bias=False)
        
        self.fusion = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
    def forward(self, x):
        # Chạy phần stem và fire modules
        stem = self.base.stem(x)
        
        # Fire 1-2
        f2 = self.base.fire1(stem)
        f2 = self.base.fire2(f2)
        f2 = self.base.se1(f2)
        
        # Fire 3-4
        p1 = self.base.pool1(f2)
        f4 = self.base.fire3(p1)
        f4 = self.base.fire4(f4)
        f4 = self.base.se2(f4)
        
        # Fire 5-6
        p2 = self.base.pool2(f4)
        f6 = self.base.fire5(p2)
        f6 = self.base.fire6(f6)
        f6 = self.base.se3(f6)
        
        # Fire 7-8
        f8 = self.base.fire7(f6)
        f8 = self.base.fire8(f8)
        f8 = self.base.se4(f8)
        
        # Lateral connections (Feature Pyramid Network style)
        lat4 = self.lateral_fire4(f4)
        lat6 = self.lateral_fire6(f6)
        lat8 = self.lateral_fire8(f8)
        
        # Upsample to match resolution
        lat6_upsampled = F.interpolate(lat6, size=lat4.shape[2:], mode='bilinear', align_corners=False)
        lat8_upsampled = F.interpolate(lat8, size=lat4.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate multi-scale features
        multi_scale_features = torch.cat([lat4, lat6_upsampled, lat8_upsampled], dim=1)
        
        # Final prediction
        x = self.fusion(multi_scale_features)
        
        return x.view(x.size(0), -1)

# Kiểm tra số tham số
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chọn model (nâng cao hoặc standard)
USE_FEATURE_PYRAMID = True  # Set to False nếu muốn dùng model cơ bản

if USE_FEATURE_PYRAMID:
    model = FeaturePyramidSqueezeNet(num_classes=7, input_channels=1).to(device)
else:
    model = EnhancedSqueezeNet(num_classes=7, input_channels=1).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Under 3M threshold: {'Yes' if total_params < 3000000 else 'No'}")

dummy_input = torch.randn(1, 1, 48, 48).to(device)
with torch.no_grad():
    model.eval()
    output = model(dummy_input)

print(f"Output shape: {output.shape}")
