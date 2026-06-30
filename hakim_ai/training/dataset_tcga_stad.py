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
        manifest = pd.read_csv(manifest_csv)
        
        # Create a mapping from patient_id to row data
        patient_data = {str(row['patient_id']): row for _, row in manifest.iterrows()}
        
        self.slides = []
        if os.path.exists(feature_dir):
            for fname in os.listdir(feature_dir):
                if not fname.endswith('.pt'):
                    continue
                # TCGA patient ID is typically the first 12 chars: TCGA-XX-XXXX
                patient_id = fname[:12]
                if patient_id in patient_data:
                    self.slides.append((fname, patient_data[patient_id]))

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fname, row = self.slides[idx]
        feature_path = os.path.join(self.feature_dir, fname)
        
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
    Collate function for MIL bags. 
    Strictly expects batch_size=1 to avoid padding memory bombs.
    """
    if len(batch) > 1:
        raise ValueError("collate_mil_bags requires batch_size=1 to avoid memory bloat. Use gradient accumulation instead.")
        
    features = batch[0][0].unsqueeze(0) # [1, N, D]
    labels_dict = batch[0][1]
    
    mask = torch.ones(1, features.shape[1], dtype=torch.bool)
    
    batched_labels = {
        k: v.unsqueeze(0) for k, v in labels_dict.items()
    }
    
    return features, mask, batched_labels
