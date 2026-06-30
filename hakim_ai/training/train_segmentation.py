"""
Training loop for segmentation model on GCHTID.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hakim_ai.config import PipelineConfig
from hakim_ai.training.dataset_gchtid import GCHTIDDataset


def train_segmentation(cfg: PipelineConfig):
    if not cfg.training.gchtid_data_root:
        print("GCHTID dataset path not configured. Skipping segmentation training.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.training.device != "auto":
        device = torch.device(cfg.training.device)
    
    dataset = GCHTIDDataset(cfg.training.gchtid_data_root, split='train')
    if len(dataset) == 0:
        print("No data found. Skipping training.")
        return
        
    safe_batch_size = min(cfg.training.batch_size, 4)
    dataloader = DataLoader(dataset, batch_size=safe_batch_size, shuffle=True, num_workers=4)
    
    try:
        from transformers import SegformerForSemanticSegmentation
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0",
            num_labels=7,
            ignore_mismatched_sizes=True
        ).to(device)
    except ImportError:
        print("transformers package is required for real segmentation training.")
        return
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(cfg.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, masks in tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}"):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=images, labels=masks)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")
        
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "segformer_gchtid.pt")
    torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    
    cfg = PipelineConfig.from_yaml(args.config)
    train_segmentation(cfg)
