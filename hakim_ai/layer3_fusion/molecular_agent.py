"""
Layer 3 — Molecular Prediction Agent.

Predicts key STAD biomarkers directly from H&E features:
  - MSI/dMMR status (MSI-H vs. MSS)
  - Lauren classification (intestinal | diffuse | mixed)
  - HER2 status (positive | negative | equivocal)
  - EBV association probability

Clinical anchor: Kather et al. 2019 (Nature Medicine) demonstrated
feasibility of H&E-based MSI prediction in gastric + colorectal cancer.
TIL density is the strongest single histological correlate of MSI-H.

Mock implementation: uses TIL density from segmentation + seeded RNG
to produce plausible, correlated predictions.
Real: fine-tuned MIL model on TCGA-STAD with multi-label supervision.
"""
from __future__ import annotations

import math
import random
from typing import Optional

from hakim_ai.foundation_models.base_encoder import BaseEncoder
from hakim_ai.config import MolecularConfig
from hakim_ai.types import (
    EBVStatus,
    EvidenceBundle,
    HER2Status,
    LaurenClassification,
    MolecularPrediction,
    MSIStatus,
)
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer3.molecular")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class MolecularPredictionAgent:
    """
    H&E → molecular biomarker prediction agent.

    Inputs:  EvidenceBundle
    Outputs: MolecularPrediction
    """

    def __init__(self, cfg: MolecularConfig, encoder: Optional[BaseEncoder] = None, seed: int = 42):
        self.cfg = cfg
        self.encoder = encoder
        self._rng = random.Random(seed)

    def run(self, evidence: EvidenceBundle) -> MolecularPrediction:
        logger.info("Molecular prediction started")

        til = evidence.segmentation.til_density
        tumour = evidence.segmentation.tumour_fraction
        features = self._aggregate_patch_features(evidence)

        # MSI prediction: high TIL density is the strongest H&E correlate
        msi_prob = self._predict_msi(til, features)
        msi_status = (
            MSIStatus.MSI_HIGH
            if msi_prob >= self.cfg.msi_threshold
            else MSIStatus.MSS
        )

        # Lauren classification
        lauren_probs = self._predict_lauren(features)
        lauren_class = max(lauren_probs, key=lauren_probs.get)  # type: ignore[arg-type]
        lauren_map = {
            "intestinal": LaurenClassification.INTESTINAL,
            "diffuse": LaurenClassification.DIFFUSE,
            "mixed": LaurenClassification.MIXED,
        }

        # HER2
        her2_prob = self._predict_her2(features, lauren_class)
        if her2_prob >= self.cfg.her2_threshold + 0.15:
            her2_status = HER2Status.POSITIVE
        elif her2_prob >= self.cfg.her2_threshold - 0.10:
            her2_status = HER2Status.EQUIVOCAL
        else:
            her2_status = HER2Status.NEGATIVE

        # EBV
        ebv_prob = self._predict_ebv(til, features)
        ebv_status = (
            EBVStatus.POSITIVE
            if ebv_prob >= self.cfg.ebv_threshold
            else EBVStatus.NEGATIVE
        )

        prediction = MolecularPrediction(
            msi_status=msi_status,
            msi_probability=round(msi_prob, 4),
            lauren_class=lauren_map.get(lauren_class, LaurenClassification.UNKNOWN),
            lauren_confidence=round(lauren_probs[lauren_class], 4),
            her2_status=her2_status,
            her2_probability=round(her2_prob, 4),
            ebv_status=ebv_status,
            ebv_probability=round(ebv_prob, 4),
            raw_logits={
                "msi_logit": round(math.log(msi_prob / (1 - msi_prob + 1e-8)), 4),
                "her2_logit": round(math.log(her2_prob / (1 - her2_prob + 1e-8)), 4),
                "ebv_logit": round(math.log(ebv_prob / (1 - ebv_prob + 1e-8)), 4),
            },
        )

        logger.info(
            "Molecular prediction: MSI=%s(%.2f) Lauren=%s HER2=%s EBV=%s(%.2f)",
            prediction.msi_status.value,
            prediction.msi_probability,
            prediction.lauren_class.value,
            prediction.her2_status.value,
            prediction.ebv_status.value,
            prediction.ebv_probability,
        )
        return prediction

    # ------------------------------------------------------------------
    # Private predictors
    # ------------------------------------------------------------------

    def _aggregate_patch_features(self, evidence: EvidenceBundle) -> dict:
        """Aggregate patch-level feature statistics for slide-level prediction."""
        patches = evidence.navigation.selected_patches
        if not patches:
            return {"mean_importance": 0.5, "max_importance": 0.5}
            
        scores = [p.importance_score for p in patches]
        
        # Real: compute mean pooled feature vector from all patches
        features = [p.feature_vector for p in patches if getattr(p, "feature_vector", None) is not None]
        mean_vector = None
        if features:
            import numpy as np
            mean_vector = np.mean(features, axis=0).tolist()
            
        return {
            "mean_importance": sum(scores) / len(scores),
            "max_importance": max(scores),
            "patch_count": len(patches),
            "mean_vector": mean_vector,
        }

    def _predict_msi(self, til: float, features: dict) -> float:
        """
        MSI probability from TIL density + importance features.
        """
        if self.encoder is not None and not getattr(self.encoder, "mock_mode", True) and features.get("mean_vector"):
            # Real: linear classifier on aggregated features
            # Mock linear weights for demonstration
            v = features["mean_vector"]
            logit = sum(x * 1.5 for x in v[:10]) + til * 3.5 - 1.0
            return round(_sigmoid(logit), 4)
            
        logit = 3.5 * til - 1.0 + self._rng.gauss(0, 0.3)
        return round(_sigmoid(logit), 4)

    def _predict_lauren(self, features: dict) -> dict[str, float]:
        """
        Lauren classification probability distribution.
        """
        if self.encoder is not None and not getattr(self.encoder, "mock_mode", True) and features.get("mean_vector"):
            v = features["mean_vector"]
            # Mock multiclass linear output
            logits = {
                "intestinal": sum(x * 1.2 for x in v[10:20]) + 0.5,
                "diffuse": sum(x * 1.2 for x in v[20:30]),
                "mixed": sum(x * 1.2 for x in v[30:40]) - 0.5,
            }
            # Softmax
            exp_logits = {k: math.exp(val) for k, val in logits.items()}
            total = sum(exp_logits.values())
            return {k: round(v / total, 4) for k, v in exp_logits.items()}
            
        base = {"intestinal": 0.55, "diffuse": 0.30, "mixed": 0.15}
        noisy = {k: max(0.01, v + self._rng.gauss(0, 0.08)) for k, v in base.items()}
        total = sum(noisy.values())
        return {k: round(v / total, 4) for k, v in noisy.items()}

    def _predict_her2(self, features: dict, lauren_class: str) -> float:
        """
        HER2 positivity probability.
        """
        base = 0.15 if lauren_class == "intestinal" else 0.07
        
        if self.encoder is not None and not getattr(self.encoder, "mock_mode", True) and features.get("mean_vector"):
            v = features["mean_vector"]
            logit = sum(x * 2.0 for x in v[40:50]) - 2.0
            score = _sigmoid(logit)
            return round((score + base) / 2.0, 4)
            
        return round(max(0.0, min(1.0, base + self._rng.gauss(0, 0.05))), 4)

    def _predict_ebv(self, til: float, features: dict) -> float:
        """
        EBV probability (~10% of gastric cancers).
        """
        base = 0.10 + til * 0.15
        
        if self.encoder is not None and not getattr(self.encoder, "mock_mode", True) and features.get("mean_vector"):
            v = features["mean_vector"]
            logit = sum(x * 2.0 for x in v[50:60]) - 2.0
            score = _sigmoid(logit)
            return round((score + base) / 2.0, 4)
            
        return round(max(0.0, min(1.0, base + self._rng.gauss(0, 0.04))), 4)