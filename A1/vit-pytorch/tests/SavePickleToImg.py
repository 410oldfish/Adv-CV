from PIL import Image
import numpy as np
import os
from config import cifar10_label_names

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_cifar10_images(data_batch_path, save_dir):
    data_dict = unpickle(data_batch_path)
    raw_images = data_dict[b'data']
    raw_filenames = data_dict[b'filenames']
    labels = data_dict[b'labels']

    os.makedirs(save_dir, exist_ok=True)

    for i in range(len(raw_images)):
        img_flat = raw_images[i]
        R = img_flat[0:1024].reshape(32, 32)
        G = img_flat[1024:2048].reshape(32, 32)
        B = img_flat[2048:].reshape(32, 32)
        img = np.dstack((R, G, B))
        img_pil = Image.fromarray(img)

        label_index = labels[i]
        label_name = cifar10_label_names[label_index]
        filename = raw_filenames[i].decode('utf-8')

        # 保存为 <类别名>_<原始文件名>.jpg
        new_filename = f"{label_name}_{filename}.jpg"
        img_pil.save(os.path.join(save_dir, new_filename))

    print(f"Saved {len(raw_images)} labeled images to: {save_dir}")
