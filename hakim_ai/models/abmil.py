"""
Attention-Based MIL (ABMIL) module.
Based on Ilse et al., 2018.
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
        class Tanh:
            pass
        class Sigmoid:
            pass
    nn = DummyNN()
    torch = None

class GatedAttentionMIL(nn.Module):
    """
    Gated attention MIL aggregator.
    Input:  (B, N, D) bag of patch features
    Output: (B, D_out) slide-level prediction, and (B, N) attention weights
    """
    def __init__(self, input_dim: int = 1536, hidden_dim: int = 256, num_heads: int = 3):
        super().__init__()
        self.L = input_dim
        self.D = hidden_dim
        self.K = num_heads

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )
        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(self.D, self.K)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, L)
        mask: (B, N) boolean mask where True means valid patch
        Returns:
            slide_embed: (B, L)
            A: (B, N) attention weights
        """
        A_V = self.attention_V(x)  # (B, N, D)
        A_U = self.attention_U(x)  # (B, N, D)
        A = self.attention_weights(A_V * A_U)  # (B, N, K)
        A = torch.softmax(A, dim=1)  # (B, N, K)
        
        # A: (B, N, K) -> (B, K, N)
        A = A.transpose(1, 2)
        # x: (B, N, L)
        # M: (B, K, N) x (B, N, L) -> (B, K, L)
        M = torch.bmm(A, x)
        
        return M, A
