"""
Layer 2 — Navigation Agent.

Mirrors the pathologist's multi-scale review workflow:
  5× overview → 20× region selection → 40× high-power patch sampling

Outputs an importance map and a ranked list of patch coordinates.

Testing implementation: samples patches from a seeded grid with synthetic
importance scores. Real implementation: uses the UNI 2 patch encoder +
an attention MIL head to compute per-patch importance scores, then
selects the top-K patches across three magnification levels.
"""
from __future__ import annotations

import math
import random
from typing import Any, List, Tuple
import os

from hakim_ai.config import NavigationConfig
from hakim_ai.foundation_models.base_encoder import BaseEncoder
from hakim_ai.models.abmil import GatedAttentionMIL
from hakim_ai.types import (
    NavigationResult,
    PatchCoordinate,
    QCResult,
    WSIData,
)
from hakim_ai.utils.image_utils import (
    compute_tissue_mask,
    extract_patch_coordinates,
    extract_patch_from_wsi,
)
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer2.navigation")

# Map magnification level name to level index in OpenSlide pyramid
_MAG_TO_LEVEL = {5: 3, 10: 2, 20: 1, 40: 0}


class NavigationAgent:
    """
    Multi-scale WSI navigation and ROI selection agent.

    Inputs:  WSIData, QCResult
    Outputs: NavigationResult
    """

    def __init__(self, cfg: NavigationConfig, encoder: BaseEncoder, seed: int = 42, normalizer: Any = None, checkpoint_dir: str = "checkpoints"):
        self.cfg = cfg
        self.encoder = encoder
        self.normalizer = normalizer
        self._rng = random.Random(seed)
        
        # Load ABMIL model for importance scoring
        self.abmil = None
        try:
            import torch
            self.device = torch.device("cuda" if torch.cuda.is_available() and getattr(encoder, "use_gpu", False) else "cpu")
            self.abmil = GatedAttentionMIL(input_dim=encoder.embedding_dim).to(self.device)
            ckpt_path = os.path.join(checkpoint_dir, "abmil_multi_task.pt")
            if os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                if 'mil' in state:
                    self.abmil.load_state_dict(state['mil'])
        except Exception as e:
            logger.warning(f"Could not load ABMIL model: {e}")

    def unload(self) -> None:
        if getattr(self, "abmil", None) is not None:
            del self.abmil
            self.abmil = None
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def run(self, wsi_data: WSIData, qc_result: QCResult) -> NavigationResult:
        logger.info("Navigation started for patient %s", wsi_data.patient_id)

        tissue_mask = compute_tissue_mask(wsi_data.thumbnail)

        all_patches: List[PatchCoordinate] = []
        
        # Calculate true downsample factor for thumbnail
        # Thumbnail is size 512, level 0 might be 100k x 100k
        thumb_downsample = 128
        if wsi_data.level_dimensions and wsi_data.thumbnail:
            try:
                import numpy as np
                w_full = wsi_data.level_dimensions[0][0]
                thumb_w = len(wsi_data.thumbnail[0]) if isinstance(wsi_data.thumbnail, list) else np.array(wsi_data.thumbnail).shape[1]
                thumb_downsample = max(1, w_full // thumb_w)
            except Exception:
                pass
                
        try:
            import torch
        except ImportError:
            pass
        # 1. Gather all patches and extract features
        extracted_patches = []
        for mag in self.cfg.magnification_levels:
            level = _MAG_TO_LEVEL.get(mag, 1)
            level = min(level, wsi_data.level_count - 1)
            
            coords = extract_patch_coordinates(
                tissue_mask=tissue_mask,
                patch_size=self.cfg.patch_size,
                thumbnail_downsample=thumb_downsample,
                top_k=max(1, self.cfg.top_k_patches // len(self.cfg.magnification_levels)),
                seed=self._rng.randint(0, 10_000),
            )
            for x, y in coords:
                patch = extract_patch_from_wsi(
                    wsi_path=wsi_data.wsi_path, x=x, y=y, level=level,
                    size=(self.cfg.patch_size, self.cfg.patch_size),
                    slide_handle=getattr(wsi_data, "slide_handle", None),
                    normalizer=self.normalizer,
                )
                try:
                    feat = self.encoder.encode_patch(patch)
                except Exception:
                    feat = [0.0] * self.encoder.embedding_dim
                extracted_patches.append({
                    "x": x, "y": y, "level": level, "feat": feat
                })
        # 2. Score patches using ABMIL
        if not extracted_patches:
            logger.warning("No tissue patches extracted. Returning empty navigation result.")
            return NavigationResult(
                selected_patches=[],
                importance_map=self._build_importance_map([], rows=16, cols=16),
                diagnostic_regions=[],
                magnification_levels_used=self.cfg.magnification_levels,
                top_patch_count=0
            )

        if self.abmil is None:
            raise RuntimeError("ABMIL model is missing, cannot score patches.")
            
        with torch.no_grad():
            # features: (1, N, D)
            feats_t = torch.tensor([p["feat"] for p in extracted_patches], dtype=torch.float32).unsqueeze(0).to(self.device)
            _, attn = self.abmil(feats_t)
            # Normalize attention to [0, 1] range for importance scores
            attn_scores = attn.view(-1)
            if len(attn_scores) > 0:
                attn_scores = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min() + 1e-8)
            attn_scores = attn_scores.cpu().numpy()
            
        for p, score in zip(extracted_patches, attn_scores):
            score = round(float(score.item()), 4)
            patch_obj = PatchCoordinate(
                x=p["x"], y=p["y"], level=p["level"],
                width=self.cfg.patch_size, height=self.cfg.patch_size,
                importance_score=score, label=self._label_region(score)
            )
            patch_obj.feature_vector = p["feat"]
            all_patches.append(patch_obj)

        # Select representative patches using KMeans clustering on feature embeddings
        # to ensure morphological diversity (phenotype representation).
        if len(all_patches) <= self.cfg.top_k_patches:
            selected = all_patches
        else:
            try:
                from sklearn.cluster import KMeans
                import numpy as np
                
                valid_patches = [p for p in all_patches if p.feature_vector is not None]
                n_clusters = min(5, self.cfg.top_k_patches)
                
                if len(valid_patches) >= n_clusters:
                    features_matrix = np.array([p.feature_vector for p in valid_patches])
                    kmeans = KMeans(n_clusters=n_clusters, random_state=self._rng.randint(0, 10000), n_init='auto')
                    clusters = kmeans.fit_predict(features_matrix)
                    
                    selected = []
                    patches_per_cluster = self.cfg.top_k_patches // n_clusters
                    
                    for c in range(n_clusters):
                        cluster_patches = [p for i, p in enumerate(valid_patches) if clusters[i] == c]
                        cluster_patches.sort(key=lambda p: p.importance_score, reverse=True)
                        selected.extend(cluster_patches[:patches_per_cluster])
                        
                    # Fill remaining slots with highest scoring unselected patches
                    selected_set = set(id(p) for p in selected)
                    all_patches.sort(key=lambda p: p.importance_score, reverse=True)
                    for p in all_patches:
                        if len(selected) >= self.cfg.top_k_patches:
                            break
                        if id(p) not in selected_set:
                            selected.append(p)
                            selected_set.add(id(p))
                else:
                    all_patches.sort(key=lambda p: p.importance_score, reverse=True)
                    selected = all_patches[: self.cfg.top_k_patches]
                    
            except ImportError:
                # Fallback if sklearn is not installed
                all_patches.sort(key=lambda p: p.importance_score, reverse=True)
                selected = all_patches[: self.cfg.top_k_patches]

        # Build a low-resolution importance map (16×16)
        importance_map = self._build_importance_map(selected, rows=16, cols=16)

        diagnostic_regions = self._identify_diagnostic_regions(selected)

        result = NavigationResult(
            selected_patches=selected,
            importance_map=importance_map,
            diagnostic_regions=diagnostic_regions,
            magnification_levels_used=self.cfg.magnification_levels,
            top_patch_count=len(selected),
        )

        logger.info(
            "Navigation done: %d patches selected across levels %s",
            len(selected),
            self.cfg.magnification_levels,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _label_region(self, score: float) -> str:
        if score >= 0.75:
            return "tumour_core"
        if score >= 0.50:
            return "invasive_front"
        return "stroma"

    def _build_importance_map(
        self, patches: List[PatchCoordinate], rows: int = 16, cols: int = 16
    ) -> List[List[float]]:
        """Create a coarse 2-D importance heat map."""
        grid = [[0.0] * cols for _ in range(rows)]
        if not patches:
            return grid
        max_x = max(p.x for p in patches) or 1
        max_y = max(p.y for p in patches) or 1
        for p in patches:
            row = min(rows - 1, int(p.y / max_y * (rows - 1)))
            col = min(cols - 1, int(p.x / max_x * (cols - 1)))
            grid[row][col] = max(grid[row][col], p.importance_score)
        return grid

    def _identify_diagnostic_regions(
        self, patches: List[PatchCoordinate]
    ) -> List[dict]:
        """Group top patches into coarse diagnostic regions."""
        if not patches:
            return []
        top = patches[:5]
        return [
            {
                "region_id": f"R{i + 1}",
                "x": p.x,
                "y": p.y,
                "importance": p.importance_score,
                "label": p.label,
                "level": p.level,
            }
            for i, p in enumerate(top)
        ]