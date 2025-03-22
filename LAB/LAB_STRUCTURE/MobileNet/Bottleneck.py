import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
from LAB.LAB_STRUCTURE.MobileNet.train import apply_quantization, apply_pruning , train,evaluate
from LAB.LAB_STRUCTURE.MobileNet.LightFERMobileNet import LightFERMobileNet
from LAB.LAB_STRUCTURE.MobileNet.Data_processing import FERDataset
# Định nghĩa khối Bottleneck
class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(Bottleneck, self).__init__()
        hidden_channels = in_channels * expansion_factor

        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=stride,
                               padding=1, groups=hidden_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
