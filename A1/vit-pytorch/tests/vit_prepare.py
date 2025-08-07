import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from vit_pytorch.efficient import ViT
from linformer import Linformer

from config import batch_size, test_batch_size

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Transforms 数据增强
train_transforms = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


# test_transforms = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
# ])

# Dataset 数据集，已经按batch划分好
class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        label = img_path.split('/')[-1].split('.')[0].split('_')[0].lower()
        label = 1 if label == 'dog' else 0
        return img, label, img_path

# Data loader
# 训练集 & 验证集
def get_train_valid_loaders(train_dir):
    train_list = glob.glob(os.path.join(train_dir, '*.jpg'))
    train_list = [p.replace('\\', '/') for p in train_list]
    labels = [p.split('/')[-1].split('.')[0].split('_')[0] for p in train_list]
    train_list, valid_list = train_test_split(train_list, test_size=0.2, stratify=labels, random_state=42)


    train_data = CatsDogsDataset(train_list, transform=train_transforms)
    valid_data = CatsDogsDataset(valid_list, transform=test_transforms)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

# 测试集
def get_test_loader(test_dir):
    test_list = glob.glob(os.path.join(test_dir, '*.jpg'))
    test_list = [p.replace('\\', '/') for p in test_list]
    test_data = CatsDogsDataset(test_list, transform=test_transforms)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)
    return test_loader


# Model
def build_model():
    efficient_transformer = Linformer(
        dim=128,
        seq_len=16+1,
        depth=12,
        heads=8,
        k=64
    )
    model = ViT(
        dim=128,
        image_size=32,
        patch_size=8,
        num_classes=2,
        transformer=efficient_transformer,
        channels=3,
    ).to(device)
    return model
