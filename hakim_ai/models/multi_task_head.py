"""
Multi-task classification head for the molecular agent.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class DummyNN:
        class Module:
            pass
        class Sequential:
            pass
        class Linear:
            pass
        class ReLU:
            pass
        class Dropout:
            pass
    nn = DummyNN()
    torch = None

class MultiTaskHead(nn.Module):
    def __init__(self, input_dim: int = 1536):
        super().__init__()
        
        # Shared representations
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Individual task heads (no HER2)
        self.msi_head = nn.Linear(512, 1)        # Binary
        self.ebv_head = nn.Linear(512, 1)        # Binary
        self.lauren_head = nn.Linear(512, 3)     # 3 classes: intestinal, diffuse, mixed

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x is now (B, 3, L) representing 3 task-specific attention heads
        
        if x.dim() == 2:
            # Fallback for single-head or inference without batch-head dim
            feat = self.shared(x)
            return {
                'msi': self.msi_head(feat),
                'ebv': self.ebv_head(feat),
                'lauren': self.lauren_head(feat)
            }
            
        msi_feat = self.shared(x[:, 0, :])
        ebv_feat = self.shared(x[:, 1, :])
        lauren_feat = self.shared(x[:, 2, :])
        
        return {
            'msi': self.msi_head(msi_feat),
            'ebv': self.ebv_head(ebv_feat),
            'lauren': self.lauren_head(lauren_feat)
        }
