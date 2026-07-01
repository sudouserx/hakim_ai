"""
UNI 2 patch encoder adapter.

Real implementation: load ViT-H weights from Hugging Face (MahmoodLab/UNI2-h)
and forward patches through the encoder.

Swap by providing the HF token via environment variable HF_TOKEN.
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Any, List

from hakim_ai.foundation_models.base_encoder import BaseEncoder





class UNI2Encoder(BaseEncoder):
    """
    Adapter for the UNI 2 ViT-H patch encoder.

    Attempts to load real weights (requires torch + transformers + valid HF_TOKEN).
    """

    EMBEDDING_DIM = 1536   # ViT-H output dimension for UNI 2

    def __init__(self, device: str = None, use_gpu: bool = True):
        self.use_gpu = use_gpu
        if device is None:
            try:
                import torch
                self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
        self._model = None

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    def load(self) -> None:
        if self._model is None:
            self._load_real_model()

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _encode_patch(self, patch: Any) -> List[float]:
        return self._real_encode_patch(patch)

    # ------------------------------------------------------------------
    # Real model path (stub — fill in with actual torch code)
    # ------------------------------------------------------------------

    def _load_real_model(self) -> None:
        """
        Load UNI 2 weights.
        Requires: pip install torch torchvision timm huggingface_hub
        """
        try:
            import torch
            import timm
            from huggingface_hub import login
            from torchvision import transforms
        except ImportError as exc:
            raise ImportError(
                "Real UNI 2 requires torch, timm, torchvision, and huggingface_hub. "
                "Install with: pip install 'hakim_ai[models]'"
            ) from exc

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
        else:
            print("Warning: HF_TOKEN not set. Model download may fail if it is gated.")

        timm_kwargs = {
            'img_size': 224,
            'patch_size': 14,
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5,
            'embed_dim': 1536,
            'mlp_ratio': 2.66667 * 2,
            'num_classes': 0,
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked,
            'act_layer': torch.nn.SiLU,
            'reg_tokens': 8,
            'dynamic_img_size': True
        }
        self._model = timm.create_model("hf-hub:MahmoodLab/uni2-h", pretrained=True, **timm_kwargs)
        self._model = self._model.eval().to(self.device)
        
        self._transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def _real_encode_patch(self, patch: Any) -> List[float]:
        """Forward pass through the real ViT-H model."""
        import torch
        if patch is None:
            return [0.0] * self.EMBEDDING_DIM
            
        if self._model is None:
            self.load()
            
        x = self._transform(patch).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            with torch.autocast(device_type=self.device if self.device != "cpu" else "cpu"):
                feat = self._model(x)
                # L2 normalize
                feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
                
        return feat.squeeze(0).cpu().tolist()
        
    def _encode_batch(self, patches: List[Any]) -> List[List[float]]:
            
        import torch
        if not patches:
            return []
            
        valid_patches = [p for p in patches if p is not None]
        if not valid_patches:
            return [[0.0] * self.EMBEDDING_DIM for _ in patches]
            
        if self._model is None:
            self.load()
            
        tensors = [self._transform(p) for p in valid_patches]
        x = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            with torch.autocast(device_type=self.device if self.device != "cpu" else "cpu"):
                feats = self._model(x)
                feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
                
        if self.device != "cpu":
            torch.cuda.empty_cache()
            
        feat_list = feats.cpu().tolist()
        
        # Map back including Nones
        result = []
        idx = 0
        for p in patches:
            if p is None:
                result.append([0.0] * self.EMBEDDING_DIM)
            else:
                result.append(feat_list[idx])
                idx += 1
                
        return result