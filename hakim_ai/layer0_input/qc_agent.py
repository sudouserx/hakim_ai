"""
Layer 0 — Quality Control Agent.

Evaluates slide quality across three dimensions:
  1. Stain quality (H&E chromogen balance via OD estimation)
  2. Focus quality (Laplacian variance proxy)
  3. Tissue coverage (tissue vs. background fraction)

If any dimension falls below the configured threshold the slide is
rejected before entering the diagnostic pipeline.
"""
from __future__ import annotations

import random
from typing import List, Optional

from hakim_ai.config import QCConfig
from hakim_ai.types import QCResult, WSIData
from hakim_ai.utils.image_utils import (
    build_normalizer,
    compute_tissue_mask,
    tissue_coverage,
    estimate_focus_quality,
    estimate_stain_quality,
    detect_artifacts,
)
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer0.qc")


class QCAgent:
    """
    Quality gate for incoming WSI slides.

    Inputs:  WSIData
    Outputs: QCResult (passed | failed with reason)
    """

    def __init__(self, cfg: QCConfig, seed: int = 42):
        self.cfg = cfg
        self._normalizer = build_normalizer(cfg.stain_normalizer)
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, wsi_data: WSIData) -> QCResult:
        logger.info("QC started for patient %s", wsi_data.patient_id)

        # 1. Stain normalization (modifies file on disk in real impl)
        normalized_path = self._normalizer.normalize(wsi_data.wsi_path)

        # 2. Quality metric estimation
        tissue_mask = compute_tissue_mask(wsi_data.thumbnail)
        cov_score = tissue_coverage(tissue_mask)
        
        # Sample patches from the thumbnail for multi-region QC assessment
        import numpy as np
        if wsi_data.thumbnail is not None:
            thumb_np = np.array(wsi_data.thumbnail)
            h, w = thumb_np.shape[:2]
            
            # Simple grid sampling on the thumbnail
            patch_size = min(32, min(h, w))
            stain_scores = []
            focus_scores = []
            artifacts = []
            
            for y in range(0, h - patch_size + 1, patch_size):
                for x in range(0, w - patch_size + 1, patch_size):
                    patch = thumb_np[y:y+patch_size, x:x+patch_size]
                    stain_scores.append(estimate_stain_quality(patch))
                    focus_scores.append(estimate_focus_quality(patch))
                    artifacts.extend(detect_artifacts(patch))
                    
            stain_score = np.mean(stain_scores) if stain_scores else estimate_stain_quality(wsi_data.thumbnail)
            focus_score = np.mean(focus_scores) if focus_scores else estimate_focus_quality(wsi_data.thumbnail)
            artifacts = list(set(artifacts))
        else:
            # If no thumbnail, use default failsafe
            stain_score = 0.5
            focus_score = 0.5
            artifacts = []

        # 3. Artifact detection
        if self.cfg.reject_on_artifact and artifacts:
            logger.warning("Artifacts detected: %s", artifacts)

        # 4. Area estimation (rough, using mpp and level-0 dimensions)
        area_mm2 = self._estimate_area_mm2(wsi_data, cov_score)

        # 5. Pass/fail decision
        raw_metrics = {
            "stain_quality_raw": stain_score,
            "focus_quality_raw": focus_score,
            "coverage_raw": cov_score,
        }

        passed, rejection_reason = self._evaluate(
            stain_score, focus_score, cov_score, artifacts
        )

        result = QCResult(
            passed=passed,
            stain_quality_score=stain_score,
            focus_quality_score=focus_score,
            coverage_score=cov_score,
            tissue_area_mm2=area_mm2,
            artifacts_detected=artifacts,
            rejection_reason=rejection_reason,
            normalized_wsi_path=normalized_path if passed else None,
            raw_metrics=raw_metrics,
        )

        if passed:
            logger.info(
                "QC passed — stain=%.2f focus=%.2f coverage=%.2f area=%.1fmm²",
                stain_score, focus_score, cov_score, area_mm2,
            )
        else:
            logger.warning("QC FAILED: %s", rejection_reason)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _evaluate(
        self,
        stain: float,
        focus: float,
        coverage: float,
        artifacts: List[str],
    ) -> tuple[bool, Optional[str]]:
        reasons: List[str] = []
        if stain < self.cfg.min_stain_quality:
            reasons.append(
                f"Stain quality {stain:.2f} < threshold {self.cfg.min_stain_quality}"
            )
        if focus < self.cfg.min_focus_quality:
            reasons.append(
                f"Focus quality {focus:.2f} < threshold {self.cfg.min_focus_quality}"
            )
        if coverage < self.cfg.min_coverage:
            reasons.append(
                f"Tissue coverage {coverage:.2f} < threshold {self.cfg.min_coverage}"
            )
        if self.cfg.reject_on_artifact and artifacts:
            reasons.append(f"Artifacts detected: {', '.join(artifacts)}")

        if reasons:
            return False, "; ".join(reasons)
        return True, None

    def _estimate_area_mm2(self, wsi_data: WSIData, coverage: float) -> float:
        """Approximate tissue area in mm²."""
        if not wsi_data.level_dimensions:
            return 0.0
        w_px, h_px = wsi_data.level_dimensions[0]
        mpp = wsi_data.mpp or 0.25
        area_total_mm2 = (w_px * mpp / 1000) * (h_px * mpp / 1000)
        return round(area_total_mm2 * coverage, 2)