"""
GCHTID Dataset Loader for training segmentation models (SegFormer/HoVerNet).

WARNING — Label provenance:
    GCHTID labels are weak proxy labels derived from cross-domain transfer
    (NCT-CRC-HE-100K colorectal annotations → gastric cancer WSIs via model
    prediction with whole-slide pathologist review, NOT per-patch annotation).
    Downstream segmentation models trained on this data should be treated as
    tile-level tissue-composition estimators, not boundary-accurate segmenters.
"""
from __future__ import annotations

import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class GCHTIDDataset(Dataset):
    """
    Gastric Cancer Histology Tissue Image Dataset (GCHTID).
    Provides image and 7-channel mask pairs for semantic segmentation.
    Classes: 0=background, 1=tumour, 2=stroma, 3=TIL, 4=necrosis,
             5=normal_gland, 6=muscle
    """
    def __init__(self, data_root: str, split: str = 'train', img_size: int = 512):
        self.img_dir = os.path.join(data_root, split, 'images')
        self.mask_dir = os.path.join(data_root, split, 'masks')
        self.img_size = img_size
        
        if os.path.exists(self.img_dir):
            self.samples = [
                f for f in os.listdir(self.img_dir) if f.endswith('.png')
            ]
        else:
            self.samples = []
            
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fname = self.samples[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        # Assuming mask is a single-channel image with class indices 0-5
        mask_np = np.array(mask.resize((self.img_size, self.img_size), Image.NEAREST))
        
        img_t = self.transform(img)
        mask_t = torch.from_numpy(mask_np).long()
        
        return img_t, mask_t
