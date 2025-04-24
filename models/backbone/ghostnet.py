import torch
import torch.nn as nn
import math

__all__ = ['create_ghostnet_micro']


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        # Sử dụng ReLU6 thay vì ReLU thông thường (tốt hơn cho lượng tử hóa)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU6(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU6(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class SEModule(nn.Module):
    """Squeeze-and-Excitation module tối ưu hóa"""

    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU6(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        # Giảm SE ratio để giảm tham số
        se_ratio = 4  # Tăng ratio để giảm kích thước

        self.ghost1 = GhostModule(inp, hidden_dim, kernel_size=1, relu=True)

        # Tích chập depthwise
        if stride > 1:
            self.conv_dw = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride,
                                     (kernel_size - 1) // 2, groups=hidden_dim, bias=False)
            self.bn_dw = nn.BatchNorm2d(hidden_dim)
        else:
            self.conv_dw = nn.Sequential()
            self.bn_dw = nn.Sequential()

        # SE module (tối ưu)
        if use_se:
            self.se = SEModule(hidden_dim, reduction=se_ratio)
        else:
            self.se = nn.Sequential()

        self.ghost2 = GhostModule(hidden_dim, oup, kernel_size=1, relu=False)

        # Skip connection
        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        residual = x

        # GhostModule 1
        x = self.ghost1(x)

        # Depthwise Conv
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # SE
        x = self.se(x)

        # GhostModule 2
        x = self.ghost2(x)

        # Shortcut
        x += self.shortcut(residual)

        return x


class GhostNetMicro(nn.Module):
    def __init__(self, width_multiplier=0.3, num_classes=3):  # Giảm width mặc định
        super(GhostNetMicro, self).__init__()
        input_channel = max(8, int(16 * width_multiplier))  # Đảm bảo tối thiểu 8 kênh

        # Giảm số kênh đầu ra
        self.last_channel = max(32, int(512 * width_multiplier))  # Đảm bảo tối thiểu 32 kênh

        # Cấu trúc mới nhỏ gọn hơn, loại bỏ một số tầng không cần thiết
        # format: in_channels, hidden_channels, out_channels, kernel_size, stride, use_se
        self.cfgs = [
            # in, hidden, out, k, s, SE
            [16, 16, 16, 3, 1, 0],  # Giữ nguyên
            [16, 48, 24, 3, 2, 0],  # Giữ nguyên
            [24, 72, 24, 3, 1, 0],  # Giữ nguyên
            [24, 72, 40, 5, 2, 1],  # Giữ nguyên
            [40, 120, 40, 5, 1, 1],  # Giữ nguyên
            # Giảm số tầng để giảm độ phức tạp
            [40, 240, 80, 3, 2, 0],
            [80, 200, 80, 3, 1, 0],
            # Loại bỏ 2 tầng ở giữa
            [80, 280, 112, 5, 2, 1],  # Kết hợp tính năng của các tầng đã loại bỏ
            [112, 480, 160, 5, 1, 1],  # Tầng cuối cùng
        ]

        # Tầng đầu tiên với kích thước nhỏ hơn
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)  # Thay ReLU bằng ReLU6
        )

        # Tạo các tầng chính
        layers = []
        for in_channel, hidden_channel, out_channel, kernel_size, stride, use_se in self.cfgs:
            in_channel = max(8, int(in_channel * width_multiplier))  # Đảm bảo tối thiểu 8 kênh
            hidden_channel = max(8, int(hidden_channel * width_multiplier))
            out_channel = max(8, int(out_channel * width_multiplier))

            layers.append(GhostBottleneck(
                in_channel, hidden_channel, out_channel, kernel_size, stride, use_se))

        # Tầng cuối nhỏ gọn hơn
        output_channel = max(16, int(self.cfgs[-1][2] * width_multiplier))
        layers.append(
            nn.Sequential(
                nn.Conv2d(output_channel, self.last_channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(self.last_channel),
                nn.ReLU6(inplace=True),
            )
        )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Thêm dropout trước lớp linear để giảm overfitting
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.last_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def create_ghostnet_micro(num_classes=3, width=0.3):
    """
    Tạo mô hình GhostNet Micro với width cho trước
    width=0.3 cho mô hình siêu nhỏ, cân bằng giữa độ chính xác và kích thước
    """
    return GhostNetMicro(width_multiplier=width, num_classes=num_classes)