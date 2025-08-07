import os
import random
import numpy as np
import torch

# Training hyperparameters
batch_size = 64
test_batch_size = 3
epochs = 20
lr = 3e-5
gamma = 0.7
seed = 42

def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    print(f"✅ Random seed set to {seed}")
