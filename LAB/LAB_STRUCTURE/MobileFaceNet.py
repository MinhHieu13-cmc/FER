import torch
import torch.nn as nn
import torch.nn.functional as F


# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# Improved Bottleneck với Dilated Convolution và SE
class ImprovedBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, dilation=1):
        super(ImprovedBottleneck, self).__init__()
        hidden_channels = in_channels * expansion_factor
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.stride = stride
        self.use_residual = (in_channels == out_channels) and (stride == 1)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.use_residual:
            out = out + x
        return out


# MobileFaceNet Mới
class NewMobileFaceNet(nn.Module):
    def __init__(self, num_classes=8):
        super(NewMobileFaceNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.bottlenecks = nn.Sequential(
            ImprovedBottleneck(32, 16, 1, 1),
            ImprovedBottleneck(16, 24, 6, 2, dilation=1),
            ImprovedBottleneck(24, 24, 6, 1),
            ImprovedBottleneck(24, 32, 6, 2, dilation=2),  # Thêm dilation
            ImprovedBottleneck(32, 32, 6, 1),
            ImprovedBottleneck(32, 64, 6, 2),
            ImprovedBottleneck(64, 64, 6, 1),
            ImprovedBottleneck(64, 96, 6, 1),
            ImprovedBottleneck(96, 160, 6, 2, dilation=2),  # Thêm dilation
            ImprovedBottleneck(160, 160, 6, 1),
            ImprovedBottleneck(160, 320, 6, 1),
        )

        self.conv2 = nn.Conv2d(320, 128, kernel_size=1, bias=False)  # Giảm từ 1280 xuống 128
        self.bn2 = nn.BatchNorm2d(128)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = F.relu6(self.bn1(self.conv1(x)))
        out = self.bottlenecks(out)
        out = F.relu6(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


# Khởi tạo và kiểm tra số tham số
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NewMobileFaceNet(num_classes=8).to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Hàm huấn luyện và validate giữ nguyên như trước
# Thêm vào đây nếu cần