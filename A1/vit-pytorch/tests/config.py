import os
import random
import numpy as np
import torch
from MyViT import ViT
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
lr = 3e-5
gamma = 0.7
seed = 42

# Model
def build_model():
    model = ViT(
        image_size=32,
        patch_size=8,
        num_classes=10,
        dim=128,
        depth = 12,
        heads = 8,
        mlp_dim = 256,
        dropout=0.1,
        emb_dropout=0.1
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
