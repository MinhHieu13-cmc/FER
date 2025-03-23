import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
# Ví dụ Dataset tùy chỉnh (giả lập AffectNet)
class FERDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # List of grayscale images (numpy arrays)
        self.labels = labels  # List of labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label