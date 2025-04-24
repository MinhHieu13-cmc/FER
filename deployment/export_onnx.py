import onnx
import torch
import argparse
import os
from collections import OrderedDict

from ..models.backbone.fer_mobile_vit import FERMobileViTDAN
from ..models.backbone.fer_iformer_dan import create_fer_iformer_dan


def export_to_onnx(model_path, output_path, model_type='mobilevit', opset_version=11,
                   batch_size=1, img_size=160, num_classes=3, num_heads=2):
    """
    Xuất một mô hình PyTorch sang định dạng ONNX

    Args:
        model_path: Đường dẫn đến file model (.pth)
        output_path: Đường dẫn xuất file ONNX
        model_type: Loại mô hình ('mobilevit' hoặc 'iformer')
        opset_version: Phiên bản ONNX opset
        batch_size: Kích thước batch
        img_size: Kích thước ảnh đầu vào
        num_classes: Số lượng lớp
        num_heads: Số heads trong DAN (cho iFormer)
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Khởi tạo mô hình dựa trên loại
    print(f"Đang tải mô hình {model_type} từ {model_path}...")

    if model_type == 'mobilevit':
        model = FERMobileViTDAN(num_classes=num_classes, num_heads=num_heads)
    elif model_type == 'iformer':
        model = create_fer_iformer_dan(num_classes=num_classes, num_heads=num_heads)
    else:
        raise ValueError(f"Loại mô hình không được hỗ trợ: {model_type}")

    # Tải trọng số từ checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Xử lý state_dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    # Kiểm tra nếu cần xử lý tiền tố 'model.'
    if any(k.startswith('model.') for k in state_dict.keys()):
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            if key.startswith('model.'):
                name_key = key[6:]  # Loại bỏ tiền tố 'model.'
                new_state_dict[name_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict

    # Tải trọng số vào mô hình
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Mô hình đã được tải thành công.")

    # Tạo input mẫu
    dummy_input = torch.randn(
        (batch_size, 3, img_size, img_size),
        device=device
    )

    print(f"==> Xuất mô hình sang định dạng ONNX tại '{output_path}'")

    # Định nghĩa tên đầu vào/đầu ra
    input_names = ["input"]
    output_names = ["output"]

    # Xuất mô hình
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        do_constant_folding=True  # Optimize model with constant folding
    )

    print(f"==> Mô hình đã xuất thành công!")

    # Tính và hiển thị kích thước file (MB)
    model_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Kích thước mô hình ONNX: {model_size:.2f} MB")


def parse_args():
    parser = argparse.ArgumentParser(description='Chuyển đổi mô hình FER sang ONNX')
    parser.add_argument('--model', type=str, required=True, help='Đường dẫn đến file checkpoint (.pth)')
    parser.add_argument('--output', type=str, required=True, help='Đường dẫn đến file ONNX đầu ra')
    parser.add_argument('--opset', type=int, default=11, help='Phiên bản ONNX opset (mặc định: 11)')
    parser.add_argument('--batch-size', type=int, default=1, help='Kích thước batch (mặc định: 1)')
    parser.add_argument('--img-size', type=int, default=160, help='Kích thước ảnh đầu vào (mặc định: 160)')
    parser.add_argument('--num-classes', type=int, default=3, help='Số lượng lớp (mặc định: 3)')
    parser.add_argument('--model-type', type=str, default='mobilevit',
                        choices=['mobilevit', 'iformer'], help='Loại mô hình (mặc định: mobilevit)')
    parser.add_argument('--num-heads', type=int, default=2, help='Số lượng attention heads (mặc định: 2)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        model_type=args.model_type,
        opset_version=args.opset,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_classes=args.num_classes,
        num_heads=args.num_heads
    )