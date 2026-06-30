"""
Training loop for router classifier.
"""
from __future__ import annotations

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from hakim_ai.training.dataset_gashistsdb import GasHisSDBDataset
from hakim_ai.foundation_models import build_patch_encoder
from hakim_ai.config import PipelineConfig


class RouterClassifier(nn.Module):
    def __init__(self, input_dim: int = 1536):
        super().__init__()
        # 3 classes: benign, suspicious, malignant
        self.fc = nn.Linear(input_dim, 3)
        
    def forward(self, x):
        return self.fc(x)


def train_router(cfg: PipelineConfig):
    if not cfg.training.gashis_data_root:
        print("GasHisSDB dataset path not configured. Skipping router training.")
        return
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cfg.training.device != "auto":
        device = torch.device(cfg.training.device)
    
    dataset = GasHisSDBDataset(cfg.training.gashis_data_root, split='train')
    if len(dataset) == 0:
        print("No data found. Skipping training.")
        return
        
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4)
    
    encoder = build_patch_encoder(cfg.foundation_models)
    
    # We only train the linear head, not the encoder
    model = RouterClassifier(input_dim=encoder.embedding_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(cfg.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}"):
            # Map Normal (0) to Benign (0), Abnormal (1) to Malignant (2)
            mapped_labels = torch.where(labels == 1, 2, 0).to(device)
            
            # Real encode using the foundation model
            from torchvision.transforms import ToPILImage
            to_pil = ToPILImage()
            pil_images = [to_pil(img) for img in images]
            features_list = []
            for img in pil_images:
                feat = encoder.encode_patch(img)
                features_list.append(torch.tensor(feat))
            features = torch.stack(features_list).to(device)
            
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, mapped_labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(dataloader):.4f}")
        
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.training.checkpoint_dir, "router_head.pt")
    torch.save(model.state_dict(), ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    
    cfg = PipelineConfig.from_yaml(args.config)
    train_router(cfg)
