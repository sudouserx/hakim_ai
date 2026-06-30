"""
Training loop for ABMIL multi-task classifier (Layer 3 Molecular Agent).
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
from hakim_ai.training.dataset_tcga_stad import TCGAStadDataset, collate_mil_bags


def train_mil(cfg: PipelineConfig):
    if not cfg.training.tcga_feature_dir or not cfg.training.tcga_manifest_csv:
        print("TCGA dataset paths not configured. Skipping MIL classifier training.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.training.device != "auto":
        device = torch.device(cfg.training.device)
    
    # Import inside function to avoid circular imports during setup
    from hakim_ai.models.abmil import GatedAttentionMIL
    from hakim_ai.models.multi_task_head import MultiTaskHead
    
    dataset = TCGAStadDataset(cfg.training.tcga_feature_dir, cfg.training.tcga_manifest_csv)
    if len(dataset) == 0:
        print("No data found. Skipping training.")
        return
        
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, collate_fn=collate_mil_bags)
    
    # Example for UNI2 dimensions (1536)
    embed_dim = 1536
    mil = GatedAttentionMIL(input_dim=embed_dim, hidden_dim=256).to(device)
    model = MultiTaskHead(input_dim=embed_dim).to(device)
    
    optimizer = torch.optim.AdamW(list(mil.parameters()) + list(model.parameters()), lr=cfg.training.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.num_epochs)
    
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()
    
    for epoch in range(cfg.training.num_epochs):
        mil.train()
        model.train()
        
        epoch_loss = 0.0
        for features, mask, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            features = features.to(device)
            mask = mask.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            optimizer.zero_grad()
            
            # Forward
            slide_embed, attention = mil(features, mask)
            logits = model(slide_embed)
            
            # Multi-task loss
            loss_msi = bce_loss(logits['msi'].squeeze(), labels['msi'])
            loss_ebv = bce_loss(logits['ebv'].squeeze(), labels['ebv'])
            loss_lauren = ce_loss(logits['lauren'], labels['lauren'])
            loss_her2 = ce_loss(logits['her2'], labels['her2'])
            
            loss = loss_msi + loss_ebv + loss_lauren + loss_her2
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        scheduler.step()
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")
        
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "abmil_multi_task.pt")
    torch.save({
        'mil': mil.state_dict(),
        'head': model.state_dict()
    }, ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    
    cfg = PipelineConfig.from_yaml(args.config)
    train_mil(cfg)
