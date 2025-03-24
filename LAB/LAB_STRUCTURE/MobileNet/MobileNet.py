import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
from LAB.LAB_STRUCTURE.MobileNet.LightFERMobileNet import LightFERMobileNet


# Tối ưu hóa
def apply_pruning(model, pruning_rate=0.3):
    for name, param in model.named_parameters():
        if 'weight' in name:
            tensor = param.data.cpu()
            mask = tensor.abs() > tensor.abs().quantile(pruning_rate)
            param.data = tensor * mask.float().to(param.device)
    return model


def apply_quantization(model):
    model.half()
    return model


# Dataset
class FERDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


# Huấn luyện
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        inputs, labels = inputs.to(device).half(), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f"Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc


# Đánh giá
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device).half(), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
    return test_loss, test_acc


# Main
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Dữ liệu giả lập
    dummy_images = [np.random.rand(224, 224) for _ in range(1000)]
    dummy_labels = np.random.randint(0, 7, size=1000)
    train_dataset = FERDataset(dummy_images, dummy_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    # Khởi tạo mô hình
    model = LightFERMobileNet(num_classes=7).to(device)
    model = apply_pruning(model, pruning_rate=0.3)
    model = apply_quantization(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, criterion, optimizer, device, epoch)
        evaluate(model, test_loader, criterion, device)

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")