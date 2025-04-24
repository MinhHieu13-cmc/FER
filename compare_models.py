import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import argparse

# Import từ dự án
from models.backbone.fer_mobile_vit import FERMobileViTDAN
from models.backbone.fer_iformer_dan import FERiFormerDAN
from data.dataloader import valid_loader


def parse_args():
    parser = argparse.ArgumentParser(description='So sánh các mô hình FER')
    parser.add_argument('--mobilevit_path', type=str, required=True, help='Đường dẫn đến mô hình MobileViT-DAN')
    parser.add_argument('--iformer_path', type=str, required=True, help='Đường dẫn đến mô hình iFormer-DAN')
    parser.add_argument('--output_dir', type=str, default='results', help='Thư mục lưu kết quả')
    parser.add_argument('--num_heads', type=int, default=2, help='Số heads trong iFormer-DAN')
    return parser.parse_args()


def evaluate_model(model, dataloader, device):
    all_preds = []
    all_labels = []
    all_probs = []

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


def main():
    args = parse_args()

    # Tạo thư mục lưu trữ
    os.makedirs(args.output_dir, exist_ok=True)

    # Thiết bị
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    # Đường dẫn lưu kết quả
    comparison_path = os.path.join(args.output_dir, "model_comparison.png")
    cm_path = os.path.join(args.output_dir, "models_confusion_matrix.png")

    # Kiểm tra mô hình tồn tại
    if not os.path.exists(args.mobilevit_path):
        print(f"Không tìm thấy mô hình MobileViT-DAN tại: {args.mobilevit_path}")
        return

    if not os.path.exists(args.iformer_path):
        print(f"Không tìm thấy mô hình iFormer-DAN tại: {args.iformer_path}")
        return

    # Tải mô hình
    print("Đang tải mô hình...")
    mobilevit_model = FERMobileViTDAN(num_classes=3)
    mobilevit_model.load_state_dict(torch.load(args.mobilevit_path, map_location='cpu'))

    iformer_model = FERiFormerDAN(num_classes=3, num_heads=args.num_heads, pretrained=False)
    iformer_model.load_state_dict(torch.load(args.iformer_path, map_location='cpu'))

    # Đánh giá mô hình
    print("Đánh giá MobileViT-DAN...")
    mobilevit_results = evaluate_model(mobilevit_model, valid_loader, device)

    print("Đánh giá iFormer-DAN...")
    iformer_results = evaluate_model(iformer_model, valid_loader, device)

    # In kết quả
    print("\n=== KẾT QUẢ SO SÁNH ===")
    print(
        f"MobileViT-DAN - Accuracy: {mobilevit_results['accuracy']:.4f}, F1 Score: {mobilevit_results['f1_score']:.4f}")
    print(f"iFormer-DAN - Accuracy: {iformer_results['accuracy']:.4f}, F1 Score: {iformer_results['f1_score']:.4f}")

    # Vẽ biểu đồ so sánh
    metrics = ['Accuracy', 'F1 Score']
    mobilevit_scores = [mobilevit_results['accuracy'], mobilevit_results['f1_score']]
    iformer_scores = [iformer_results['accuracy'], iformer_results['f1_score']]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width / 2, mobilevit_scores, width, label='MobileViT-DAN')
    plt.bar(x + width / 2, iformer_scores, width, label='iFormer-DAN')

    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('So sánh hiệu suất giữa MobileViT-DAN và iFormer-DAN')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()

    for i, v in enumerate(mobilevit_scores):
        plt.text(i - width / 2, v + 0.01, f'{v:.4f}', ha='center')

    for i, v in enumerate(iformer_scores):
        plt.text(i + width / 2, v + 0.01, f'{v:.4f}', ha='center')

    plt.tight_layout()
    plt.savefig(comparison_path)
    print(f"✓ Biểu đồ so sánh đã được lưu tại {comparison_path}")

    # Confusion Matrix
    class_names = ['Negative', 'Neutral', 'Positive']

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    cm_mobilevit = confusion_matrix(mobilevit_results['labels'], mobilevit_results['predictions'])
    sns.heatmap(cm_mobilevit, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('MobileViT-DAN Confusion Matrix')

    plt.subplot(1, 2, 2)
    cm_iformer = confusion_matrix(iformer_results['labels'], iformer_results['predictions'])
    sns.heatmap(cm_iformer, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('iFormer-DAN Confusion Matrix')

    plt.tight_layout()
    plt.savefig(cm_path)
    print(f"✓ Biểu đồ confusion matrix đã được lưu tại {cm_path}")


if __name__ == "__main__":
    main()