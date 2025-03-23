import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
# Tối ưu hóa: Cắt tỉa và Lượng tử hóa
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


# Hàm Huấn luyện
def train(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch}"):
        inputs, labels = inputs.to(device).half(), labels.to(device)  # FP16
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


# Hàm Đánh giá
def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device).half(), labels.to(device)  # FP16
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