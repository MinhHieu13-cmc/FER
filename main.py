import torch
import os
import time


def check_gpu_info():
    """Kiểm tra và hiển thị thông tin GPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    if device.type == 'cuda':
        # Hiển thị thông tin GPU
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
        print(f"Số lượng GPU khả dụng: {torch.cuda.device_count()}")

        # Hiển thị thông tin bộ nhớ
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB
        print(f"Tổng bộ nhớ GPU: {gpu_mem:.2f} GB")
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"Bộ nhớ GPU đã cấp phát: {allocated:.2f} GB")
        cached = torch.cuda.memory_reserved(0) / (1024 ** 3)
        print(f"Bộ nhớ GPU đã đặt trước: {cached:.2f} GB")

        return gpu_mem

    return 0


def main():
    # Thiết lập cấu hình CUDA phù hợp với PyTorch 2.0.0
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'

    # Thiết lập các tham số trực tiếp ở đây
    config = {
        'mode': 'train',  # 'train', 'train_ensemble', 'evaluate', 'predict', 'export'
        'model': 'iformer',  # 'mobilevit', 'iformer'
        'pretrained': True,  # Sử dụng pretrained weights cho iFormer
        'num_heads': 2,  # Số heads trong DAN (cho iFormer)
        'epochs': 30,  # Tăng số epochs để đạt hiệu suất tốt hơn
        'lr': 0.0001,  # Learning rate
        'batch_size': 32,  # Batch size
        'model_path': 'fer_iformer_dan_model.pth',  # Đường dẫn đến file mô hình
        'image_path': None,  # Đường dẫn đến ảnh cần dự đoán
        'output': 'fer_model.onnx',  # Đường dẫn cho file xuất
        'save_dir': 'results/models',  # Thư mục lưu mô hình
    }

    # Tạo thư mục lưu trữ
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Kiểm tra thông tin GPU tại đầu chương trình
    print("\n=== KIỂM TRA THIẾT BỊ ===")
    gpu_mem = check_gpu_info()
    print("=========================\n")

    # Giải phóng bộ nhớ GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Import dataloaders
    from data.dataloader import get_dataloaders

    # Sử dụng tăng cường dữ liệu mạnh mẽ hơn để cải thiện độ chính xác
    train_loader, valid_loader = get_dataloaders(
        batch_size=config['batch_size'],
        num_workers=4  # Tăng workers nếu có đủ RAM
    )

    if config['mode'] == 'train':
        if config['model'] == 'mobilevit':
            # Huấn luyện mô hình MobileViT-DAN
            from models.backbone.fer_mobile_vit import FERMobileViTDAN
            import torch.nn as nn
            import torch.optim as optim
            from training.train import train_main

            print("Bắt đầu quá trình huấn luyện mô hình MobileViT-DAN...")
            model = FERMobileViTDAN(num_classes=3)

            # Tính toán và hiển thị số tham số của mô hình
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Số lượng tham số của mô hình: {num_params:,}")

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['epochs'], eta_min=1e-6
            )

            model, history = train_main(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=config['epochs']
            )

            # Lưu mô hình
            save_path = os.path.join(config['save_dir'], 'fer_mobilevit_dan_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, save_path)

            print(f"Mô hình đã được lưu tại: {save_path}")

        elif config['model'] == 'iformer':
            # Huấn luyện mô hình iFormer-DAN
            from models.backbone.fer_iformer_dan import create_fer_iformer_dan
            import torch.nn as nn
            import torch.optim as optim
            from training.iformer_train import train_iformer_dan

            print(f"Bắt đầu quá trình huấn luyện mô hình iFormer-DAN với {config['num_heads']} heads...")

            # Sử dụng cấu trúc mô hình được thiết kế tối ưu cho độ chính xác cao
            # và giữ kích thước dưới 5MB
            # Mô hình này sẽ lớn hơn nhưng vẫn trong giới hạn 5MB
            embed_dims = [48, 96, 192]  # Kích thước kênh ở mức trung bình
            depths = [2, 2, 6]  # Sâu hơn ở tầng cuối để trích xuất đặc trưng tốt hơn

            print(f"Sử dụng cấu trúc tối ưu: embed_dims={embed_dims}, depths={depths}")

            model = create_fer_iformer_dan(
                num_classes=3,
                pretrained=config['pretrained'],
                num_heads=config['num_heads'],
                embed_dims=embed_dims,
                depths=depths
            )

            # Đặt mô hình sang GPU
            if torch.cuda.is_available():
                model = model.cuda()

            # Tính toán và hiển thị số tham số của mô hình
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Số lượng tham số của mô hình: {num_params:,}")
            model_size_mb = num_params * 4 / (1024 * 1024)
            print(f"Ước tính kích thước mô hình: {model_size_mb:.2f} MB")

            # Nếu mô hình có kích thước > 5MB, thử giảm kích thước
            if model_size_mb > 5.0:
                print("Điều chỉnh kích thước mô hình để giữ dưới 5MB...")
                embed_dims = [40, 80, 160]  # Giảm kích thước kênh
                depths = [2, 2, 5]  # Giảm số block ở tầng cuối

                model = create_fer_iformer_dan(
                    num_classes=3,
                    pretrained=config['pretrained'],
                    num_heads=config['num_heads'],
                    embed_dims=embed_dims,
                    depths=depths
                )

                if torch.cuda.is_available():
                    model = model.cuda()

                # Tính lại kích thước mô hình
                num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Số tham số mô hình điều chỉnh: {num_params:,}")
                model_size_mb = num_params * 4 / (1024 * 1024)
                print(f"Kích thước mô hình điều chỉnh: {model_size_mb:.2f} MB")

            # Thiết lập huấn luyện tối ưu
            # Sử dụng Label Smoothing để tăng khả năng tổng quát hóa
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

            # Sử dụng AdamW với weight decay thấp hơn
            optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=5e-5)

            # Sử dụng lịch trình học tốt hơn: Cosine Annealing với warm-up
            # Giả sử bạn có lớp scheduler này, nếu không thì dùng ReduceLROnPlateau
            try:
                from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
                scheduler = CosineAnnealingWarmRestarts(
                    optimizer, T_0=5, T_mult=2, eta_min=1e-6
                )
            except:
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=3, verbose=True
                )

            # Đường dẫn lưu mô hình
            model_save_path = os.path.join(config['save_dir'], f"fer_iformer_dan_h{config['num_heads']}_model.pth")

            # Tiến hành huấn luyện
            model, history = train_iformer_dan(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=config['epochs'],
                batch_size=config['batch_size'],
                model_save_path=model_save_path,
                img_dir='results',
                use_mixed_precision=True  # Tăng tốc huấn luyện với mixed precision
            )

            # Lưu checkpoint đầy đủ
            checkpoint_path = os.path.join(config['save_dir'], f"fer_iformer_dan_h{config['num_heads']}_checkpoint.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, checkpoint_path)

            print(f"Mô hình đã được lưu tại: {model_save_path}")
            print(f"Checkpoint đã được lưu tại: {checkpoint_path}")

    elif config['mode'] == 'train_ensemble':
        # Huấn luyện mô hình ensemble
        from models.ensemble import run_cross_validation_and_ensemble
        ensemble_model = run_cross_validation_and_ensemble()
        print("Đã hoàn thành huấn luyện ensemble")

    elif config['mode'] == 'evaluate':
        # Đánh giá mô hình
        if config['model'] == 'mobilevit':
            from models.backbone.fer_mobile_vit import FERMobileViTDAN
            model = FERMobileViTDAN(num_classes=3)
        elif config['model'] == 'iformer':
            from models.backbone.fer_iformer_dan import create_fer_iformer_dan
            model = create_fer_iformer_dan(num_classes=3, num_heads=config['num_heads'])
        else:
            print("Loại mô hình không được hỗ trợ")
            return

        from evaluation.evaluate import evaluate_model

        # Tải trọng số
        checkpoint = torch.load(config['model_path'],
                                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        evaluate_model(model, valid_loader)

    elif config['mode'] == 'predict':
        # Dự đoán ảnh
        from deployment.inference import FERDeployment

        deployment = FERDeployment(model_path=config['model_path'], model_type=config['model'],
                                   num_heads=config['num_heads'])
        if config['image_path']:
            result = deployment.predict_image(config['image_path'], visualize=True)
            print(f"Dự đoán: {result['predicted_emotion']} ({result['confidence']:.2f}%)")
        else:
            print("Vui lòng cung cấp đường dẫn ảnh với tham số image_path")

    elif config['mode'] == 'export':
        # Xuất mô hình sang ONNX
        from deployment.export_onnx import export_to_onnx

        if os.path.exists(config['model_path']):
            export_to_onnx(config['model_path'], config['output'], model_type=config['model'],
                           num_heads=config['num_heads'])
        else:
            print(f"Không tìm thấy file mô hình tại: {config['model_path']}")

    # Hiển thị thông tin GPU sau khi hoàn thành
    if torch.cuda.is_available():
        print("\n=== THÔNG TIN GPU SAU KHI HOÀN THÀNH ===")
        allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
        print(f"Bộ nhớ GPU đã cấp phát: {allocated:.2f} GB")
        max_allocated = torch.cuda.max_memory_allocated(0) / (1024 ** 3)
        print(f"Bộ nhớ GPU tối đa đã cấp phát: {max_allocated:.2f} GB")
        print("=========================================\n")


if __name__ == '__main__':
    main()