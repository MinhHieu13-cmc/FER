import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .ghostnet import create_ghostnet_micro


class SimplifiedAttention(nn.Module):
    """
    Module chú ý đơn giản hóa, tránh lỗi chia nhóm
    """

    def __init__(self, in_features, reduction=8):
        super().__init__()
        # Giảm số kênh để xử lý
        mid_features = max(8, in_features // 4)

        # Spatial Attention đơn giản hóa
        self.spatial = nn.Sequential(
            nn.Conv2d(in_features, mid_features, kernel_size=1),
            nn.BatchNorm2d(mid_features),
            nn.ReLU6(inplace=True),
            nn.Conv2d(mid_features, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Channel Attention đơn giản hóa
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.channel = nn.Sequential(
            nn.Linear(in_features, max(4, in_features // reduction)),
            nn.ReLU6(inplace=True),
            nn.Linear(max(4, in_features // reduction), in_features),
            nn.Sigmoid()
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Spatial Attention
        spatial_weight = self.spatial(x)
        x_spatial = x * spatial_weight

        # Channel Attention
        batch_size = x.size(0)
        channel_weight = self.gap(x_spatial).view(batch_size, -1)
        channel_weight = self.channel(channel_weight).view(batch_size, -1, 1, 1)

        # Kết hợp
        out = x_spatial * channel_weight

        return out


class FERMobileViTDAN(nn.Module):
    def __init__(self, num_classes=3, width=0.3, num_heads=1):
        super(FERMobileViTDAN, self).__init__()

        # Sử dụng GhostNet với kích thước nhỏ gọn
        self.ghostnet = create_ghostnet_micro(num_classes=num_classes, width=width)

        # Lấy phần xương sống của mô hình
        self.conv_stem = self.ghostnet.conv_stem
        self.features = self.ghostnet.features

        # Tính toán số kênh đầu ra với dummy input
        dummy_input = torch.randn(1, 3, 160, 160)
        with torch.no_grad():
            x = self.conv_stem(dummy_input)
            x = self.features(x)
            self.out_channels = x.shape[1]
            print(f"Backbone output channels: {self.out_channels}")

        # Sử dụng module chú ý đơn giản hóa để tránh lỗi
        self.attention_heads = nn.ModuleList([
            SimplifiedAttention(self.out_channels, reduction=4) for _ in range(num_heads)
        ])

        # Global pooling và classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        # Backbone feature extraction
        x = self.conv_stem(x)
        features = self.features(x)

        # Áp dụng nhiều attention head khi cần
        if len(self.attention_heads) > 1:
            # Multi-head processing
            head_outputs = []
            for head in self.attention_heads:
                head_outputs.append(head(features))

            # Trung bình các đầu ra
            combined = torch.stack(head_outputs).mean(dim=0)
        else:
            # Single-head processing
            combined = self.attention_heads[0](features)

        # Global pooling
        x = self.avgpool(combined)
        x = torch.flatten(x, 1)

        # Dropout và classification
        x = self.dropout(x)
        x = self.fc(x)

        return x


# Hàm helper để tạo mô hình
def create_fer_model(num_classes=3, width=0.3, num_heads=1):
    """
    Tạo mô hình FER với cấu trúc tối ưu:
    - width: Hệ số nhân kích thước (0.3 cho mô hình siêu nhỏ <1MB)
    - num_heads: Số lượng attention head (1-2 là đủ)
    """
    return FERMobileViTDAN(num_classes=num_classes, width=width, num_heads=num_heads)