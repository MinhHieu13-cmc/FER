import torch
import torch.nn as nn
import torch.nn.functional as F
from LAB.LAB_STRUCTURE.MobileNet.Bottleneck import Bottleneck  # Nhập Bottleneck từ file riêng


class LightFERMobileNet(nn.Module):
    def __init__(self, num_classes=7):
        super(LightFERMobileNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.bottlenecks = nn.Sequential(
            Bottleneck(32, 16, 1, 1),
            Bottleneck(16, 24, 6, 2), Bottleneck(24, 24, 6, 1),
            Bottleneck(24, 32, 6, 2), Bottleneck(32, 32, 6, 1), Bottleneck(32, 32, 6, 1),
            Bottleneck(32, 64, 6, 2), Bottleneck(64, 64, 6, 1), Bottleneck(64, 64, 6, 1), Bottleneck(64, 64, 6, 1),
            Bottleneck(64, 96, 6, 1), Bottleneck(96, 96, 6, 1), Bottleneck(96, 96, 6, 1),
            Bottleneck(96, 160, 6, 2), Bottleneck(160, 160, 6, 1), Bottleneck(160, 160, 6, 1),
            Bottleneck(160, 320, 6, 1),
        )
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = self.bottlenecks(x)
        x = F.relu6(self.bn2(self.conv2(x)))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x