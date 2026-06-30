"""
TCGA-STAD Dataset Loader for MIL Bag Training.
Loads pre-extracted patch features (.pt or .npy) and ground truth labels.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset


class TCGAStadDataset(Dataset):
    """
    Multi-Instance Learning dataset for TCGA-STAD slides.
    Returns (bag_of_features, labels_dict) per slide.
    """
    def __init__(self, feature_dir: str, manifest_csv: str):
        self.feature_dir = feature_dir
        self.manifest = pd.read_csv(manifest_csv)
        # Filter to slides that have extracted features
        self.slides = [
            row for _, row in self.manifest.iterrows()
            if os.path.exists(os.path.join(feature_dir, f"{row['patient_id']}.pt"))
        ]

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        row = self.slides[idx]
        patient_id = row['patient_id']
        feature_path = os.path.join(self.feature_dir, f"{patient_id}.pt")
        
        # Load features: shape (num_patches, embedding_dim)
        features = torch.load(feature_path, weights_only=True)
        
        # Parse labels
        labels = {
            'msi': torch.tensor(1.0 if row['msi_status'] == 'MSI-H' else 0.0, dtype=torch.float32),
            'ebv': torch.tensor(1.0 if row['ebv_status'] == 'EBV' else 0.0, dtype=torch.float32),
            'lauren': torch.tensor(self._encode_lauren(row['lauren_class']), dtype=torch.long),
            'her2': torch.tensor(self._encode_her2(row['her2_status']), dtype=torch.long),
        }
        
        return features, labels

    def _encode_lauren(self, val: str) -> int:
        mapping = {'intestinal': 0, 'diffuse': 1, 'mixed': 2}
        return mapping.get(str(val).lower(), 0)

    def _encode_her2(self, val: str) -> int:
        mapping = {'negative': 0, 'equivocal': 1, 'positive': 2}
        return mapping.get(str(val).lower(), 0)


def collate_mil_bags(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Collate function to pad variable-length feature bags.
    Returns: padded_features, attention_mask, batched_labels
    """
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    
    # Get max bag size
    max_len = max(f.shape[0] for f in features)
    embed_dim = features[0].shape[1]
    
    padded_features = torch.zeros(len(features), max_len, embed_dim)
    mask = torch.zeros(len(features), max_len, dtype=torch.bool)
    
    for i, f in enumerate(features):
        l = f.shape[0]
        padded_features[i, :l, :] = f
        mask[i, :l] = True
        
    batched_labels = {
        k: torch.stack([l[k] for l in labels])
        for k in labels[0].keys()
    }
    
    return padded_features, mask, batched_labels
