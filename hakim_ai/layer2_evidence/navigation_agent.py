"""
Layer 2 — Navigation Agent.

Mirrors the pathologist's multi-scale review workflow:
  5× overview → 20× region selection → 40× high-power patch sampling

Outputs an importance map and a ranked list of patch coordinates.

Mock implementation: samples patches from a seeded grid with synthetic
importance scores. Real implementation: uses the UNI 2 patch encoder +
an attention MIL head to compute per-patch importance scores, then
selects the top-K patches across three magnification levels.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

from hakim_ai.config import NavigationConfig
from hakim_ai.foundation_models.base_encoder import BaseEncoder
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

    def __init__(self, cfg: NavigationConfig, encoder: BaseEncoder, seed: int = 42):
        self.cfg = cfg
        self.encoder = encoder
        self._rng = random.Random(seed)

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
                
        for mag in self.cfg.magnification_levels:
            level = _MAG_TO_LEVEL.get(mag, 1)
            # Ensure level is within bounds
            level = min(level, wsi_data.level_count - 1)
            
            coords = extract_patch_coordinates(
                tissue_mask=tissue_mask,
                patch_size=self.cfg.patch_size,
                thumbnail_downsample=thumb_downsample,
                top_k=max(3, self.cfg.top_k_patches // len(self.cfg.magnification_levels)),
                seed=self._rng.randint(0, 10_000),
            )
            for x, y in coords:
                score, feat = self._compute_importance(wsi_data, x, y, level)
                patch = PatchCoordinate(
                    x=x, y=y,
                    level=level,
                    width=self.cfg.patch_size,
                    height=self.cfg.patch_size,
                    importance_score=score,
                    label=self._label_region(score),
                )
                patch.feature_vector = feat
                all_patches.append(patch)

        # Sort by importance, keep top_k
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

    def _compute_importance(
        self, wsi_data: WSIData, x: int, y: int, level: int
    ) -> tuple[float, List[float]]:
        """
        Compute per-patch importance score.

        Real: encode patch with UNI 2 → linear probe for cancer/importance.
        Mock: use coordinate hash + Gaussian noise for determinism.
        """
        if getattr(self.encoder, "mock_mode", True):
            seed_val = hash((wsi_data.patient_id, x, y, level)) & 0xFFFFFFFF
            self._rng.seed(seed_val)
            base = self._rng.betavariate(2.0, 1.5)
            # Create a mock vector
            feat = self.encoder.encode_patch(None)
            return round(min(1.0, max(0.0, base)), 4), feat

        patch = extract_patch_from_wsi(
            wsi_path=wsi_data.wsi_path,
            x=x, y=y, level=level,
            size=(self.cfg.patch_size, self.cfg.patch_size),
            slide_handle=getattr(wsi_data, "slide_handle", None)
        )
        
        try:
            feat = self.encoder.encode_patch(patch)
            # Mock linear probe on the extracted feature vector
            # Real implementation would have an ABMIL attention head or prototype similarity
            score = sum(f * 0.5 for f in feat[:20]) + 0.5
            return round(min(1.0, max(0.0, score)), 4), feat
        except Exception:
            return 0.5, [0.0] * self.encoder.embedding_dim

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