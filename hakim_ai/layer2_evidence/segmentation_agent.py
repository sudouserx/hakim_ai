"""
Layer 2 — Segmentation Agent.

Delineates tissue compartments across the WSI:
  tumour · stroma · TIL (tumour-infiltrating lymphocytes) · necrosis ·
  normal glands · background

Used by:
  - Molecular prediction agent (TIL density → MSI prediction)
  - Explanation agent (evidence citations)
  - Report agent (quantitative findings section)

Mock: returns plausible fractions computed from the WSI dimensions
and a deterministic RNG. Real: HoVerNet, CellViT, or a SegFormer model
fine-tuned on GCHTID/NCT-CRC-HE-100K.
"""
from __future__ import annotations

import random
from typing import Dict, List

from hakim_ai.config import SegmentationConfig
from hakim_ai.types import (
    NavigationResult,
    TissueSegmentation,
    WSIData,
)
from hakim_ai.utils.image_utils import extract_patch_from_wsi
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer2.segmentation")


class SegmentationAgent:
    """
    Tissue compartment mapping agent.

    Inputs:  WSIData, NavigationResult
    Outputs: TissueSegmentation
    """

    def __init__(self, cfg: SegmentationConfig, seed: int = 42):
        self.cfg = cfg
        self._rng = random.Random(seed)

    def run(
        self, wsi_data: WSIData, navigation: NavigationResult
    ) -> TissueSegmentation:
        logger.info("Segmentation started for patient %s", wsi_data.patient_id)

        # In a real system: run HoVerNet or CellViT on selected patches
        # and aggregate compartment masks to slide level.
        fracs = self._estimate_fractions(wsi_data)
        tme_profile = self._build_tme_profile(fracs)
        region_labels = self._annotate_regions(navigation, wsi_data)

        result = TissueSegmentation(
            tumour_fraction=fracs["tumour"],
            stroma_fraction=fracs["stroma"],
            til_density=fracs["til"],
            necrosis_fraction=fracs["necrosis"],
            normal_gland_fraction=fracs["normal_gland"],
            tme_profile=tme_profile,
            region_labels=region_labels,
        )

        logger.info(
            "Segmentation done: tumour=%.2f stroma=%.2f TIL=%.2f necrosis=%.2f",
            result.tumour_fraction,
            result.stroma_fraction,
            result.til_density,
            result.necrosis_fraction,
        )
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_fractions(self, wsi_data: WSIData) -> Dict[str, float]:
        """
        Estimate tissue compartment fractions.
        Real: perform HSV thresholding for stroma vs tumour approximation, 
        and dark nuclei thresholding for lymphocytes on thumbnail.
        """
        import numpy as np
        
        if wsi_data.thumbnail is None or isinstance(wsi_data.thumbnail, list):
            # Mock fallback
            seed_val = hash(wsi_data.patient_id) & 0xFFFFFFFF
            rng = random.Random(seed_val)
            base = {"tumour": 0.45, "stroma": 0.30, "til": 0.10, "necrosis": 0.05, "normal_gland": 0.08, "background": 0.02}
            noisy = {k: max(0.0, v + rng.gauss(0, 0.05)) for k, v in base.items()}
            total = sum(noisy.values())
            return {k: round(v / total, 4) for k, v in noisy.items()}
            
        thumb_np = np.array(wsi_data.thumbnail)
        if len(thumb_np.shape) != 3 or thumb_np.shape[2] < 3:
            return {"tumour": 0.4, "stroma": 0.3, "til": 0.1, "necrosis": 0.1, "normal_gland": 0.1, "background": 0.0}
            
        import matplotlib.colors as colors
        hsv = colors.rgb_to_hsv(thumb_np[:, :, :3] / 255.0)
        
        # Simple color-based heuristics
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        background_mask = (v > 0.9) & (s < 0.1)
        tissue_mask = ~background_mask
        
        if not np.any(tissue_mask):
            return {"tumour": 0.4, "stroma": 0.3, "til": 0.1, "necrosis": 0.1, "normal_gland": 0.1, "background": 0.0}
            
        # Pink/red is often stroma/eosinophilic
        stroma_mask = tissue_mask & (h > 0.8) & (h < 1.0) & (s > 0.2)
        # Purple/blue is often cellular/nuclei/tumour
        cellular_mask = tissue_mask & (h > 0.6) & (h <= 0.8)
        
        # Very dark dense purple = lymphocytes
        til_mask = cellular_mask & (v < 0.4)
        tumour_mask = cellular_mask & (v >= 0.4)
        
        total_pixels = thumb_np.shape[0] * thumb_np.shape[1]
        
        fracs = {
            "tumour": float(np.sum(tumour_mask)) / total_pixels,
            "stroma": float(np.sum(stroma_mask)) / total_pixels,
            "til": float(np.sum(til_mask)) / total_pixels,
            "necrosis": 0.05,  # hard to estimate from simple color
            "normal_gland": 0.05, # hard to estimate
            "background": float(np.sum(background_mask)) / total_pixels,
        }
        
        # Normalize sum to 1
        total = sum(fracs.values())
        if total > 0:
            fracs = {k: round(v / total, 4) for k, v in fracs.items()}
            
        return fracs

    def _build_tme_profile(self, fracs: Dict[str, float]) -> Dict[str, float]:
        """Return the tumour microenvironment profile."""
        return {
            "tumour_purity": fracs["tumour"],
            "stromal_fraction": fracs["stroma"],
            "immune_fraction": fracs["til"],
            "necrotic_fraction": fracs["necrosis"],
            "til_density_score": round(fracs["til"] / max(fracs["tumour"], 0.01), 3),
        }

    def _annotate_regions(
        self, navigation: NavigationResult, wsi_data: WSIData = None
    ) -> List[Dict]:
        """Attach tissue class labels to diagnostic regions."""
        import numpy as np
        annotations = []
        for region in navigation.diagnostic_regions:
            importance = region.get("importance", 0.0)
            tissue_class = (
                "tumour_stroma_interface"
                if importance > 0.6
                else "tumour_bulk"
                if importance > 0.4
                else "stroma"
            )
            
            til_density = 0.1
            if wsi_data is not None:
                patch = extract_patch_from_wsi(
                    wsi_data.wsi_path, region["x"], region["y"], region.get("level", 0),
                    size=(256, 256), slide_handle=getattr(wsi_data, "slide_handle", None)
                )
                if patch is not None and not isinstance(patch, list):
                    import matplotlib.colors as colors
                    img = np.array(patch)
                    if len(img.shape) == 3 and img.shape[2] >= 3:
                        hsv = colors.rgb_to_hsv(img[:, :, :3] / 255.0)
                        h, v = hsv[:, :, 0], hsv[:, :, 2]
                        # Dark purple nuclei approximation
                        til_mask = (h > 0.6) & (h <= 0.8) & (v < 0.4)
                        til_density = float(np.mean(til_mask)) * 2.0 # Adjust scale
                        
            annotations.append({
                **region,
                "tissue_class": tissue_class,
                "til_density_local": round(min(1.0, til_density), 3),
            })
        return annotations