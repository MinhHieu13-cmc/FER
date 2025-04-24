import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import copy
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import os


def train_main(model, train_loader, valid_loader, criterion, optimizer, scheduler,
               num_epochs=25, batch_size=None, model_save_path=None, img_dir=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Kiểm tra xem có đang sử dụng GPU không
    print(f"Sử dụng thiết bị: {device}")
    if device.type == 'cuda':
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
        print(f"Bộ nhớ GPU đã cấp phát: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
        print(f"Bộ nhớ GPU tối đa được cấp phát: {torch.cuda.max_memory_allocated(0) / 1024 ** 2:.2f} MB")
        print(f"Số lượng GPU khả dụng: {torch.cuda.device_count()}")

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }

    # Đảm bảo thư mục lưu trữ tồn tại
    if img_dir:
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(os.path.join(img_dir, 'img'), exist_ok=True)
        os.makedirs(os.path.join(img_dir, 'models'), exist_ok=True)

    since = time.time()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        # Sử dụng tqdm để hiển thị tiến độ
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for inputs, labels in train_loader_tqdm:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            # Cập nhật thanh tiến độ
            current_loss = loss.item()
            current_acc = (preds == labels).float().mean().item()
            train_loader_tqdm.set_postfix(loss=current_loss, acc=current_acc)

        # Hiển thị thông tin GPU sau mỗi epoch
        if device.type == 'cuda':
            print(f"Bộ nhớ GPU đã cấp phát: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
            print(f"Bộ nhớ GPU tối đa được cấp phát: {torch.cuda.max_memory_allocated(0) / 1024 ** 2:.2f} MB")

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        valid_loader_tqdm = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]")

        with torch.no_grad():
            for inputs, labels in valid_loader_tqdm:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

                # Cập nhật thanh tiến độ
                current_loss = loss.item()
                current_acc = (preds == labels).float().mean().item()
                valid_loader_tqdm.set_postfix(loss=current_loss, acc=current_acc)

        val_loss = val_loss / len(valid_loader.dataset)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, average='weighted')

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}')

        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            # Lưu mô hình
            if model_save_path:
                best_model_path = model_save_path
            else:
                if img_dir:
                    best_model_path = os.path.join(img_dir, "models", "best_mobilevit_model.pth")
                else:
                    best_model_path = "results/models/best_mobilevit_model.pth"
                    # Đảm bảo thư mục tồn tại
                    os.makedirs("results/models", exist_ok=True)

            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Mô hình tốt nhất đã được lưu tại: {best_model_path}")

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Plot training history
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    if img_dir:
        history_path = os.path.join(img_dir, "img", "mobilevit_training_history.png")
    else:
        history_path = "results/img/mobilevit_training_history.png"
        # Đảm bảo thư mục tồn tại
        os.makedirs("results/img", exist_ok=True)

    plt.savefig(history_path)
    print(f"✓ Biểu đồ training đã được lưu tại: {history_path}")
    plt.show()

    # Plot confusion matrix
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (MobileViT-DAN)')

    if img_dir:
        cm_path = os.path.join(img_dir, "img", "mobilevit_confusion_matrix.png")
    else:
        cm_path = "results/img/mobilevit_confusion_matrix.png"

    plt.savefig(cm_path)
    print(f"✓ Confusion matrix đã được lưu tại: {cm_path}")
    plt.show()

    return model, history


# Hàm main có thể được sử dụng để chạy trực tiếp script này
def main():
    from models.backbone.fer_mobile_vit import FERMobileViTDAN
    import torch.optim as optim
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from data.dataloader import get_dataloaders

    # Thiết lập các tham số
    batch_size = 32
    num_epochs = 10
    lr = 0.0001

    # Tạo thư mục kết quả
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/img", exist_ok=True)

    # Tải dữ liệu
    train_loader, valid_loader = get_dataloaders(batch_size=batch_size)

    # Khởi tạo mô hình
    model = FERMobileViTDAN(num_classes=3)

    # Tổng số tham số
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Số lượng tham số của mô hình: {num_params:,}")

    # Khởi tạo criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    # Đường dẫn lưu mô hình
    model_save_path = "results/models/fer_mobilevit_dan_model.pth"

    # Huấn luyện mô hình
    model, history = train_main(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        batch_size=batch_size,
        model_save_path=model_save_path,
        img_dir="results"
    )

    # Lưu checkpoint đầy đủ
    checkpoint_path = "results/models/fer_mobilevit_dan_checkpoint.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, checkpoint_path)

    print(f"Mô hình đã được lưu tại: {model_save_path}")
    print(f"Checkpoint đã được lưu tại: {checkpoint_path}")


if __name__ == "__main__":
    main()