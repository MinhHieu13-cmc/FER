import torch
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random


class ImageFolderDataset(Dataset):
    def __init__(self, root_folders, transform=None):
        """
        Khởi tạo dataset từ thư mục

        Args:
            root_folders: Dict của đường dẫn thư mục và nhãn tương ứng
                          Ví dụ: {'/path/to/Negative': 0, '/path/to/Neutral': 1, '/path/to/Positive': 2}
            transform: Transform áp dụng cho ảnh
        """
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

        # Thu thập đường dẫn ảnh và nhãn
        for folder, label in root_folders.items():
            if isinstance(label, str) and label in self.class_map:
                numeric_label = self.class_map[label]
            else:
                numeric_label = label

            # Kiểm tra thư mục có tồn tại không
            if not os.path.exists(folder):
                print(f"Thư mục không tồn tại: {folder}")
                continue

            # Lấy tất cả file ảnh trong thư mục
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files = glob.glob(os.path.join(folder, ext))
                for img_path in image_files:
                    self.image_paths.append(img_path)
                    self.labels.append(numeric_label)

        # In thông tin dataset
        # print(f"Đã tải {len(self.image_paths)} ảnh từ {len(root_folders)} thư mục")
        # print(f"Phân bố nhãn: {self._count_labels()}")

    def _count_labels(self):
        """Đếm số lượng mẫu cho mỗi nhãn"""
        counts = {}
        for label in self.labels:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return counts

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]  # Khai báo biến local thay vì global
            label = self.labels[idx]

            # Mở ảnh
            image = Image.open(img_path).convert('RGB')

            # Áp dụng transform
            if self.transform:
                image = self.transform(image)

            return image, label

        except Exception as e:
            # print(f"Lỗi khi đọc ảnh tại index {idx}: {str(e)}")
            # Tạo ảnh giả khi có lỗi
            dummy_img = Image.new('RGB', (260, 260), color='gray')
            if self.transform:
                dummy_img = self.transform(dummy_img)
            return dummy_img, 0  # Trả về nhãn mặc định


# Định nghĩa transforms
train_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Thư mục chứa dữ liệu
train_folders = {
    r'C:\Users\GIGABYTE\PycharmProjects\FER\DATASET\Data\Negative': 0,
    r'C:\Users\GIGABYTE\PycharmProjects\FER\DATASET\Data\Neutral': 1,
    r'C:\Users\GIGABYTE\PycharmProjects\FER\DATASET\Data\Positive': 2
}

valid_folders = {
    r'C:\Users\GIGABYTE\PycharmProjects\FER\DATASET\Valid\Negative': 0,
    r'C:\Users\GIGABYTE\PycharmProjects\FER\DATASET\Valid\Neutral': 1,
    r'C:\Users\GIGABYTE\PycharmProjects\FER\DATASET\Valid\Positive': 2
}

# Biến để kiểm tra xem đã tạo dataloader hay chưa
_loaders_initialized = False
train_loader = None
valid_loader = None


# Hàm để khởi tạo dataloaders
def get_dataloaders(batch_size=32, num_workers=4, force_reload=False):
    global _loaders_initialized, train_loader, valid_loader

    if not _loaders_initialized or force_reload:
        # Tạo dataset từ thư mục
        train_dataset = ImageFolderDataset(train_folders, transform=train_transform)
        valid_dataset = ImageFolderDataset(valid_folders, transform=valid_transform)

        # Tạo DataLoader
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        _loaders_initialized = True
        # print("DataLoader đã được tạo thành công!")

    return train_loader, valid_loader


# Khởi tạo dataloaders khi import module
train_loader, valid_loader = get_dataloaders()