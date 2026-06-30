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
        
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_mil_bags)
    
    # Example for UNI2 dimensions (1536)
    embed_dim = 1536
    mil = GatedAttentionMIL(input_dim=embed_dim, hidden_dim=256).to(device)
    model = MultiTaskHead(input_dim=embed_dim).to(device)
    
    optimizer = torch.optim.AdamW(list(mil.parameters()) + list(model.parameters()), lr=cfg.training.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.num_epochs)
    
    # Define Focal Loss function for binary classification
    def focal_loss_bce(logits, targets, alpha=0.25, gamma=2.0):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        loss = alpha * (1 - pt) ** gamma * bce
        return loss.mean()
        
    ce_loss = nn.CrossEntropyLoss()
    grad_accum_steps = cfg.training.batch_size if cfg.training.batch_size > 1 else 32
    
    for epoch in range(cfg.training.num_epochs):
        mil.train()
        model.train()
        epoch_loss = 0.0
        
        optimizer.zero_grad()
        for step, (features, mask, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")):
            features = features.to(device)
            mask = mask.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            # Forward
            slide_embed, attention = mil(features, mask)
            logits = model(slide_embed)
            
            # Multi-task loss (focal loss for rare classes MSI and EBV)
            loss_msi = focal_loss_bce(logits['msi'].view(-1), labels['msi'].float().view(-1))
            loss_ebv = focal_loss_bce(logits['ebv'].view(-1), labels['ebv'].float().view(-1))
            loss_lauren = ce_loss(logits['lauren'], labels['lauren'])
            
            loss = (loss_msi + loss_ebv + loss_lauren) / grad_accum_steps
            
            loss.backward()
            
            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * grad_accum_steps
            
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
