import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Squeeze-and-Excitation Block (được tối ưu)
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=8):  # Giảm reduction ratio để tăng độ phức tạp
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # Sửa lỗi cú pháp
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Depthwise Separable Convolution
class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DWConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = F.relu6(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return F.relu6(x)

# MobileNetV2 Inverted Residual Block
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # Expansion
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        
        # Linear pointwise
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
        
        # Thêm SE block sau mỗi khối để tăng hiệu suất
        self.se = SEBlock(out_channels)
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.se(self.conv(x))
        else:
            return self.se(self.conv(x))

# Transformer Block được tối ưu hóa
class LiteTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2):
        super(LiteTransformerBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        
        # MLP
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_features),
            nn.GELU(),  # GELU thường hoạt động tốt hơn ReLU với Transformer
            nn.Dropout(0.1),
            nn.Linear(hidden_features, dim),
            nn.Dropout(0.1)
        )
        
        # Thêm thông tin vị trí cho hiệu quả tốt hơn
        self.pos_embedding = nn.Parameter(torch.zeros(1, 16, dim))  # Giả sử sequence length tối đa là 16
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Thêm positional embedding
        # Cắt hoặc pad positional embedding để phù hợp với chiều dài thực tế
        pos_emb = self.pos_embedding[:, :N, :]
        x = x + pos_emb
        
        # Self-attention
        x_norm = self.norm1(x)
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Tính attention scores
        attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        x_attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        
        # Skip connection
        x = x + x_attn
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x

# EfficientFormer cải tiến
class EfficientFormer(nn.Module):
    def __init__(self, num_classes=7, embed_dim=192, depth=3, input_channels=1):  # Tăng embed_dim để tăng độ phức tạp
        super(EfficientFormer, self).__init__()
        
        # Stem: Xử lý đầu vào
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            DWConv(32, 64, stride=2),  # Giảm kích thước để tăng tốc
        )
        
        # MobileNet stages với các khối Inverted Residual
        self.stage1 = nn.Sequential(
            InvertedResidual(64, 96, stride=2, expand_ratio=4),
            InvertedResidual(96, 96, stride=1, expand_ratio=4),
        )
        
        self.stage2 = nn.Sequential(
            InvertedResidual(96, embed_dim, stride=2, expand_ratio=4),
            InvertedResidual(embed_dim, embed_dim, stride=1, expand_ratio=4),
        )
        
        # Patch embedding để chuyển từ CNN sang Transformer
        self.patch_embed = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU6(inplace=True),
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            LiteTransformerBlock(embed_dim, num_heads=8)  # Tăng số heads
            for _ in range(depth)
        ])
        
        # Global attention pooling
        self.global_pool = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
        # Khởi tạo trọng số
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # CNN feature extraction
        x = self.stem(x)           # [B, 64, H/4, W/4]
        x = self.stage1(x)         # [B, 96, H/8, W/8]
        x = self.stage2(x)         # [B, embed_dim, H/16, W/16]
        
        # Extract features
        x = self.patch_embed(x)    # [B, embed_dim, H/16, W/16]
        
        # Chuyển từ spatial features sang sequence
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        # Global pooling
        x = self.global_pool(x).mean(dim=1)  # [B, C]
        
        # Classification
        x = self.classifier(x)
        
        return x

# Kiểm tra mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EfficientFormer(num_classes=7, input_channels=1).to(device)

# In số tham số
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
print(f"Under 3M threshold: {'Yes' if total_params < 3000000 else 'No'}")

# Kiểm tra với tensor ngẫu nhiên
dummy_input = torch.randn(1, 1, 48, 48).to(device)  # Ví dụ kích thước ảnh 48x48
with torch.no_grad():
    model.eval()
    output = model(dummy_input)

print(f"Output shape: {output.shape}")
