import os
import random
import numpy as np
import torch
from MyViT import ViT
from MyCrossViT import CrossViT
#from linformer import Linformer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CIFAR-10 label names
cifar10_label_names = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck',
}

#dataset path
train_dataset_path = "data/train"
test_dataset_path = "data/test"

# Training hyperparameters
batch_size = 64
test_batch_size = 64
epochs = 100
lr = 3e-5 #1e-4
gamma = 0.7
seed = 42

# Model
# def build_model():
#     model = ViT(
#         image_size=32,
#         patch_size=8,
#         num_classes=10,
#         dim=128,
#         depth = 12,
#         heads = 8,
#         mlp_dim = 256,
#         dropout=0.1
#         ).to(device)
#     return model

def build_model():
    model = CrossViT(
        image_size=32, #图片大小
        num_classes=10, #类别
        sm_dim = 64, #小patch dim
        lg_dim = 128, # 大patch dim
        sm_patch_size=4, #小patch大小
        sm_enc_depth=2, # 小patch transformer层数
        sm_enc_heads=8, # 小patch head 数量
        sm_enc_mlp_dim=256, # 小patch MLP维度
        sm_enc_dim_head=32, # 小patch head维度
        lg_patch_size=8, #大patch大小
        lg_enc_depth=4, # 大patch transformer层数
        lg_enc_heads=8, # 大patch head 数量
        lg_enc_mlp_dim=512, # 大patch MLP维度
        lg_enc_dim_head=64, # 大patch head维度
        cross_attn_depth=2, # cross-attention层数
        cross_attn_heads=8, # cross-attention head 数量
        cross_attn_dim_head=64, # cross-attention head维度
        depth=3, # transformer层数
        dropout=0.1,
        emb_dropout=0.1,
        channels=3
    ).to(device)
    return model

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"✅ Random seed set to {seed}")
