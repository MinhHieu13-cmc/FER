import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LiteEfficientAttention(nn.Module):
    """Module chú ý hiệu quả và mạnh mẽ cho iFormer"""

    def __init__(self, dim, head_dim=32, qkv_bias=True, reduction=2):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = max(1, dim // head_dim)  # Số đầu tối ưu dựa trên kích thước kênh
        self.scale = head_dim ** -0.5

        # Kích thước kênh cho Q, K
        qk_dim = max(16, dim // reduction)

        # Ánh xạ Q, K, V với tỷ lệ nén hiệu quả
        self.q = nn.Conv2d(dim, qk_dim, kernel_size=1, bias=qkv_bias)
        self.k = nn.Conv2d(dim, qk_dim, kernel_size=1, bias=qkv_bias)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        # Ánh xạ đầu ra với attention residual
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.BatchNorm2d(dim)
        )

        # Thêm squeeze-excitation module để tăng hiệu suất
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B, C, H, W = x.shape

        # Ánh xạ Q, K, V
        q = self.q(x)  # B, C', H, W
        k = self.k(x)  # B, C', H, W
        v = self.v(x)  # B, C, H, W

        # Reshape để tính attention
        q_flat = q.reshape(B, -1, H * W).permute(0, 2, 1)  # B, H*W, C'
        k_flat = k.reshape(B, -1, H * W)  # B, C', H*W
        v_flat = v.reshape(B, -1, H * W)  # B, C, H*W

        # Tính attention và áp dụng lên values
        attn = (q_flat @ k_flat) * self.scale  # B, H*W, H*W
        attn = F.softmax(attn, dim=-1)
        x_flat = v_flat @ attn.transpose(-2, -1)  # B, C, H*W

        # Reshape về 2D
        x_2d = x_flat.reshape(B, C, H, W)

        # Áp dụng squeeze-excitation trước khi projection
        se_weight = self.se(x_2d)
        x_2d = x_2d * se_weight

        # Projection cuối cùng
        out = self.proj(x_2d)

        return out


class LiteFeedForward(nn.Module):
    """Module Feed-Forward tinh chỉnh cho hiệu suất cao"""

    def __init__(self, in_dim, hidden_dim=None, out_dim=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim

        self.net = nn.Sequential(
            # Tích chập pointwise với BN
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_layer(inplace=True),
            nn.Dropout(drop),
            # Depthwise conv để tăng khả năng học
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            act_layer(inplace=True),
            # Tích chập pointwise để ánh xạ về kích thước đầu ra
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.net(x)


class iFormerLiteBlock(nn.Module):
    """iFormer Block được tối ưu hóa cho độ chính xác cao"""

    def __init__(self, dim, head_dim=32, mlp_ratio=4., drop=0.1, drop_path=0.,
                 act_layer=nn.ReLU, layer_scale_init_value=1e-5):
        super().__init__()

        # Chuẩn hóa
        self.norm1 = nn.BatchNorm2d(dim)

        # Module chú ý
        self.attn = LiteEfficientAttention(dim, head_dim=head_dim)

        # Chuẩn hóa thứ 2
        self.norm2 = nn.BatchNorm2d(dim)

        # MLP với depthwise conv ở giữa
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LiteFeedForward(in_dim=dim, hidden_dim=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Hệ số tỷ lệ cho layer
        self.layer_scale = layer_scale_init_value > 0
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

        # DropPath để tăng khả năng tổng quát hóa
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        # Attention path với residual
        x_attn = self.norm1(x)
        x_attn = self.attn(x_attn)

        if self.layer_scale:
            x_attn = x_attn * self.gamma1.view(1, -1, 1, 1)

        x = x + self.drop_path(x_attn)

        # FFN path với residual
        x_mlp = self.norm2(x)
        x_mlp = self.mlp(x_mlp)

        if self.layer_scale:
            x_mlp = x_mlp * self.gamma2.view(1, -1, 1, 1)

        x = x + self.drop_path(x_mlp)

        return x


# Triển khai DropPath
class DropPath(nn.Module):
    """Dropout path được triển khai để tăng khả năng tổng quát hóa."""

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class LiteStem(nn.Module):
    """Stem được tinh chỉnh cho hiệu suất cao"""

    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        # Sử dụng kiến trúc 3 tầng để tăng trích xuất đặc trưng
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Thêm một lớp depthwise conv để tăng trích xuất đặc trưng không gian
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1,
                      groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.stem(x)


class FERiFormerLite(nn.Module):
    """iFormer tinh chỉnh cho FER với độ chính xác cao và kích thước nhỏ gọn"""

    def __init__(self,
                 in_chans=3,
                 num_classes=3,
                 depths=[2, 2, 6],  # Tăng độ sâu của tầng cuối
                 dims=[48, 96, 192],  # Kích thước kênh hợp lý
                 drop_path_rate=0.2,  # Tăng droppath để chống overfitting
                 layer_scale_init_value=1e-5,
                 head_dim=32,
                 num_heads=2):  # Số đầu cho DANs
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(depths)

        # Stem bao gồm các tầng đầu tiên
        self.stem = LiteStem(in_chans, dims[0])

        # Xây dựng các tầng
        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(self.num_stages):
            stage = nn.Sequential(
                *[iFormerLiteBlock(
                    dim=dims[i],
                    head_dim=head_dim,
                    mlp_ratio=4.,  # Tăng mlp_ratio cho khả năng học tốt hơn
                    drop=0.1,
                    drop_path=dpr[cur + j],
                    layer_scale_init_value=layer_scale_init_value)
                    for j in range(depths[i])]
            )

            self.stages.append(stage)

            # Transition với stride=2
            if i < self.num_stages - 1:
                transition = nn.Sequential(
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(dims[i + 1]),
                    nn.ReLU(inplace=True)
                )
                self.stages.append(transition)

            cur += depths[i]

        # Module chú ý đa đầu (DAN)
        self.num_heads = num_heads
        if num_heads > 1:
            self.dan_heads = nn.ModuleList([
                LiteEfficientAttention(dims[-1], head_dim=head_dim)
                for _ in range(num_heads)
            ])
            # Thêm attention fusion module
            self.fusion = nn.Sequential(
                nn.Conv2d(dims[-1] * num_heads, dims[-1], kernel_size=1, bias=False),
                nn.BatchNorm2d(dims[-1]),
                nn.ReLU(inplace=True)
            )

        # Pooling và phân loại
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.BatchNorm2d(dims[-1])
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(dims[-1], num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        # Xử lý đa đầu (DAN)
        if hasattr(self, 'dan_heads') and self.num_heads > 1:
            head_outputs = []
            for head in self.dan_heads:
                head_outputs.append(head(x))

            # Cách 1: Kết hợp các đầu ra bằng trung bình
            # x = torch.stack(head_outputs).mean(dim=0)

            # Cách 2: Fusion các đầu ra bằng convolution
            x = torch.cat(head_outputs, dim=1)
            x = self.fusion(x)

        # Chuẩn hóa cuối cùng
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_fer_iformer_dan(num_classes=3, pretrained=False, num_heads=2, embed_dims=None, depths=None):
    """
    Tạo mô hình iFormer-DAN tối ưu cho nhận dạng cảm xúc khuôn mặt

    Args:
        num_classes: Số lớp cần phân loại
        pretrained: Có tải trọng số pretrained không
        num_heads: Số đầu chú ý cho DAN
        embed_dims: Danh sách kích thước kênh cho các tầng, mặc định [48, 96, 192]
        depths: Danh sách số khối cho mỗi tầng, mặc định [2, 2, 6]

    Returns:
        Mô hình FERiFormerLite
    """
    # Cấu hình mặc định được tối ưu hóa cho độ chính xác cao và kích thước dưới 5MB
    if embed_dims is None:
        embed_dims = [48, 96, 192]  # Kích thước kênh cân bằng giữa độ chính xác và kích thước mô hình

    if depths is None:
        depths = [2, 2, 6]  # Đủ sâu cho độ chính xác cao, tập trung vào tầng cuối

    model = FERiFormerLite(
        in_chans=3,
        num_classes=num_classes,
        depths=depths,
        dims=embed_dims,
        drop_path_rate=0.2,
        layer_scale_init_value=1e-5,
        head_dim=32,
        num_heads=num_heads
    )

    # Tính và in kích thước mô hình
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Số lượng tham số của mô hình: {num_params:,}")
    model_size_mb = num_params * 4 / (1024 * 1024)  # Xấp xỉ kích thước theo MB (float32)
    print(f"Ước tính kích thước mô hình: {model_size_mb:.2f} MB")

    return model