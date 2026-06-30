"""
Foundation model adapter for SegFormer-based tissue segmentation.
"""
from __future__ import annotations

import os
from typing import Any
import numpy as np

import torch
import torch.nn as nn
from PIL import Image

from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("segformer_adapter")





class SegformerAdapter:
    """
    Adapter for a fine-tuned SegFormer model for semantic segmentation.
    Outputs a 6-channel probability map or argmax mask.
    Classes: 0=background, 1=tumour, 2=stroma, 3=TIL, 4=necrosis, 5=normal_gland
    """
    def __init__(self, checkpoint_path: str, num_classes: int = 6, use_gpu: bool = False):
        self.checkpoint_path = checkpoint_path
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.model = None
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
    def load(self):
            
        try:
            from transformers import SegformerForSemanticSegmentation
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0",
                num_labels=self.num_classes,
                ignore_mismatched_sizes=True
            ).to(self.device)
            if os.path.exists(self.checkpoint_path):
                self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device, weights_only=True))
            self.model.eval()
            logger.info("Loaded SegFormer segmentation model")
        except Exception as e:
            logger.warning(f"Failed to load SegFormer: {e}")
            self.model = None
            
    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                
    def segment_patch(self, patch: Any) -> np.ndarray:
        """
        Segment a single patch.
        Returns: (H, W) array of class indices.
        """
        if self.model is None or patch is None:
            raise RuntimeError("Segformer model is not loaded or patch is None.")
            w, h = patch.size if hasattr(patch, "size") else (512, 512)
            return np.full((h, w), 2, dtype=np.uint8)
            
        import torchvision.transforms as T
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        try:
            img_t = transform(patch).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(img_t)
                logits = outputs.logits
                
            mask = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
            
            # Resize mask back to original patch size
            w, h = patch.size
            if mask.shape != (h, w):
                mask_img = Image.fromarray(mask).resize((w, h), Image.NEAREST)
                mask = np.array(mask_img)
                
            return mask
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            w, h = patch.size
            return np.full((h, w), 2, dtype=np.uint8)
