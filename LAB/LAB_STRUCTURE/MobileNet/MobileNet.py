from LAB.LAB_STRUCTURE.MobileNet.Bottleneck import Bottleneck
from LAB.LAB_STRUCTURE.MobileNet.LightFERMobileNet import LightFERMobileNet
from LAB.LAB_STRUCTURE.MobileNet.train import apply_quantization, apply_pruning , train,evaluate
from LAB.LAB_STRUCTURE.MobileNet.Data_processing import FERDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

# Khởi tạo và chạy
if __name__ == "__main__":
    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tiền xử lý dữ liệu
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Giả lập dữ liệu (thay bằng AffectNet thực tế)
    import numpy as np

    dummy_images = [np.random.rand(224, 224) for _ in range(1000)]
    dummy_labels = np.random.randint(0, 7, size=1000)
    train_dataset = FERDataset(dummy_images, dummy_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Khởi tạo mô hình
    model = LightFERMobileNet(num_classes=7).to(device)
    model = apply_pruning(model, pruning_rate=0.3)
    model = apply_quantization(model)

    # Cấu hình huấn luyện
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    # Huấn luyện và đánh giá
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, criterion, optimizer, device, epoch)
        evaluate(model, train_loader, criterion, device)  # Dùng train_loader để demo

    # In số tham số
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")