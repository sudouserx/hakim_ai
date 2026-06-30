"""
GasHisSDB Dataset Loader for patch-level encoder fine-tuning and router training.
"""
from __future__ import annotations

import os
from PIL import Image
from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T


class GasHisSDBDataset(Dataset):
    """
    GasHisSDB dataset for binary patch classification (Normal vs Abnormal).
    Used for training the router linear probe.
    """
    def __init__(self, data_root: str, split: str = 'train', img_size: int = 224):
        self.data_root = os.path.join(data_root, split)
        self.img_size = img_size
        self.samples = []
        
        # Structure: root/train/Normal/, root/train/Abnormal/
        for label_name, label_idx in [('Normal', 0), ('Abnormal', 1)]:
            dir_path = os.path.join(self.data_root, label_name)
            if os.path.exists(dir_path):
                for fname in os.listdir(dir_path):
                    if fname.endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(dir_path, fname), label_idx))
                        
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip() if split == 'train' else T.Lambda(lambda x: x),
            T.RandomVerticalFlip() if split == 'train' else T.Lambda(lambda x: x),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
