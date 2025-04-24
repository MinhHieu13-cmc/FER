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
from torch.cuda.amp import autocast, GradScaler


def train_iformer_dan(model, train_loader, valid_loader, criterion, optimizer, scheduler,
                      num_epochs=25, batch_size=None, model_save_path=None, img_dir=None,
                      use_mixed_precision=True, accumulation_steps=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo GradScaler cho mixed precision training
    scaler = GradScaler() if use_mixed_precision and device.type == 'cuda' else None

    # Kiểm tra xem có đang sử dụng GPU không
    print(f"Sử dụng thiết bị: {device}")
    if device.type == 'cuda':
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
        print(f"Bộ nhớ GPU đã cấp phát: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
        print(f"Bộ nhớ GPU tối đa được cấp phát: {torch.cuda.max_memory_allocated(0) / 1024 ** 3:.2f} GB")
        print(f"Bộ nhớ GPU đã dành ra: {torch.cuda.memory_reserved(0) / 1024 ** 3:.2f} GB")
        print(f"Số lượng GPU khả dụng: {torch.cuda.device_count()}")

    # Cấu hình tối ưu hóa CUDA nếu sử dụng GPU
    if device.type == 'cuda':
        # Tối ưu hóa CUDA
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        print("Đã bật tối ưu hóa CUDA benchmark")

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

    # Ghi lại tốc độ huấn luyện và thời gian ước tính
    samples_per_sec_history = []

    # Giải phóng bộ nhớ cache trước khi bắt đầu
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []

        optimizer.zero_grad()  # Đảm bảo gradient được reset ở đầu epoch

        train_start = time.time()
        batch_count = 0
        sample_count = 0

        # Sử dụng tqdm để hiển thị tiến độ
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")

        for i, (inputs, labels) in enumerate(train_loader_tqdm):
            inputs = inputs.to(device, non_blocking=True)  # non_blocking giúp tăng tốc chuyển dữ liệu
            labels = labels.to(device, non_blocking=True)

            batch_size_current = inputs.size(0)
            sample_count += batch_size_current
            batch_count += 1

            # Mixed precision training
            if use_mixed_precision and device.type == 'cuda':
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss = loss / accumulation_steps  # Normalize loss for gradient accumulation

                # Scale loss và thực hiện backward
                scaler.scale(loss).backward()

                # Thực hiện optimizer step sau mỗi accumulation_steps
                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                # Huấn luyện thông thường (precision đầy đủ)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

                loss.backward()

                if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

            # Thu thập dự đoán và nhãn
            with torch.no_grad():
                _, preds = torch.max(outputs, 1)
                train_loss += loss.item() * batch_size_current * accumulation_steps
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            # Cập nhật thanh tiến độ
            current_loss = loss.item() * accumulation_steps
            current_acc = (preds == labels).float().mean().item()
            train_loader_tqdm.set_postfix(loss=current_loss, acc=current_acc)

            # Xóa các tensor không cần thiết để giải phóng bộ nhớ
            del inputs, labels, outputs, preds
            if i % 10 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()

        # Tính tốc độ huấn luyện
        train_time = time.time() - train_start
        samples_per_sec = sample_count / train_time
        samples_per_sec_history.append(samples_per_sec)

        # Hiển thị thông tin GPU sau mỗi epoch
        if device.type == 'cuda':
            print(f"Bộ nhớ GPU đã cấp phát: {torch.cuda.memory_allocated(0) / 1024 ** 3:.2f} GB")
            print(f"Bộ nhớ GPU tối đa được cấp phát: {torch.cuda.max_memory_allocated(0) / 1024 ** 3:.2f} GB")
            print(f"Tốc độ huấn luyện: {samples_per_sec:.2f} samples/giây")
            remaining_epochs = num_epochs - epoch - 1
            est_remaining_time = remaining_epochs * train_time / 60
            print(f"Thời gian ước tính còn lại: {est_remaining_time:.1f} phút")

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f}')
        print(f'Thời gian huấn luyện 1 epoch: {train_time / 60:.2f} phút')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []

        valid_loader_tqdm = tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Valid]")

        with torch.no_grad():
            for inputs, labels in valid_loader_tqdm:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                # Validation giai đoạn có thể sử dụng mixed precision mà không cần GradScaler
                if use_mixed_precision and device.type == 'cuda':
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
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

                # Xóa các tensor không cần thiết
                del inputs, labels, outputs, preds

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
                    best_model_path = os.path.join(img_dir, "models", "best_iformer_model.pth")
                else:
                    best_model_path = "results/models/best_iformer_model.pth"
                    # Đảm bảo thư mục tồn tại
                    os.makedirs("results/models", exist_ok=True)

            torch.save(model.state_dict(), best_model_path)
            print(f"✓ Mô hình tốt nhất đã được lưu tại: {best_model_path}")

        # Giải phóng bộ nhớ GPU sau mỗi epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    print(f'Tốc độ huấn luyện trung bình: {np.mean(samples_per_sec_history):.2f} samples/giây')

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
        history_path = os.path.join(img_dir, "img", "iformer_training_history.png")
    else:
        history_path = "results/img/iformer_training_history.png"
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
    plt.title('Confusion Matrix (iFormer-DAN)')

    if img_dir:
        cm_path = os.path.join(img_dir, "img", "iformer_confusion_matrix.png")
    else:
        cm_path = "results/img/iformer_confusion_matrix.png"

    plt.savefig(cm_path)
    print(f"✓ Confusion matrix đã được lưu tại: {cm_path}")
    plt.show()

    return model, history


# Hàm main có thể được sử dụng để chạy trực tiếp script này
def main():
    from models.backbone.fer_iformer_dan import create_fer_iformer_dan
    import torch.optim as optim
    from data.dataloader import get_dataloaders

    # Thiết lập các tham số (đã giảm để tăng tốc)
    batch_size = 16  # Giảm batch_size để giảm bộ nhớ sử dụng
    num_epochs = 10
    lr = 0.0001
    num_heads = 1  # Giảm heads từ 2 xuống 1

    # Tạo thư mục kết quả
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/img", exist_ok=True)

    # Tải dữ liệu với cấu hình tối ưu
    train_loader, valid_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=4,  # Tăng số worker để tăng tốc đọc dữ liệu
        pin_memory=True  # Giúp tăng tốc chuyển dữ liệu lên GPU
    )

    # Tạo mô hình với kích thước nhỏ hơn để tăng tốc
    small_embed_dims = [32, 64, 128]  # Giảm kích thước mô hình
    small_depths = [2, 2, 3]  # Giảm số layers

    model = create_fer_iformer_dan(
        num_classes=3,
        pretrained=True,
        num_heads=num_heads,
        embed_dims=small_embed_dims,
        depths=small_depths
    )

    # Tổng số tham số
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Số lượng tham số của mô hình: {num_params:,}")

    # Khởi tạo criterion, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # Đường dẫn lưu mô hình
    model_save_path = f"results/models/fer_iformer_dan_h{num_heads}_model.pth"

    # Huấn luyện mô hình với mixed precision và gradient accumulation
    model, history = train_iformer_dan(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        batch_size=batch_size,
        model_save_path=model_save_path,
        img_dir="results",
        use_mixed_precision=True,  # Bật mixed precision để tăng tốc
        accumulation_steps=2  # Sử dụng gradient accumulation
    )

    # Lưu checkpoint đầy đủ
    checkpoint_path = f"results/models/fer_iformer_dan_h{num_heads}_checkpoint.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, checkpoint_path)

    print(f"Mô hình đã được lưu tại: {model_save_path}")
    print(f"Checkpoint đã được lưu tại: {checkpoint_path}")


if __name__ == "__main__":
    main()
