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

Testing implementation: uses TIL density from segmentation + seeded RNG
to produce plausible, correlated predictions.
Real: fine-tuned MIL model on TCGA-STAD with multi-label supervision.
"""
from __future__ import annotations

import math
import random
import os
from typing import Optional

from hakim_ai.foundation_models.base_encoder import BaseEncoder
from hakim_ai.models.abmil import GatedAttentionMIL
from hakim_ai.models.multi_task_head import MultiTaskHead
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
        
        # Load models
        self.abmil = None
        self.head = None
        if encoder is not None:
            try:
                import torch
                self.device = torch.device("cuda" if torch.cuda.is_available() and getattr(encoder, "use_gpu", False) else "cpu")
                self.abmil = GatedAttentionMIL(input_dim=encoder.embedding_dim).to(self.device)
                self.head = MultiTaskHead(input_dim=encoder.embedding_dim).to(self.device)
                ckpt_path = "checkpoints/abmil_multi_task.pt"
                if os.path.exists(ckpt_path):
                    state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
                    if 'mil' in state:
                        self.abmil.load_state_dict(state['mil'])
                    if 'head' in state:
                        self.head.load_state_dict(state['head'])
                self.abmil.eval()
                self.head.eval()
            except ImportError:
                logger.warning("Torch not available; molecular models disabled.")
            except Exception as e:
                logger.warning(f"Could not load molecular models: {e}")

    def run(self, evidence: EvidenceBundle) -> MolecularPrediction:
        logger.info("Molecular prediction started")

        til = evidence.segmentation.til_density
        tumour = evidence.segmentation.tumour_fraction
        features = self._aggregate_patch_features(evidence)

        if self.abmil is None or self.head is None:
            raise RuntimeError("Molecular models (abmil, head) are required for prediction.")
            
        import torch
        with torch.no_grad():
            patches = evidence.navigation.selected_patches
            feats = [p.feature_vector for p in patches if getattr(p, "feature_vector", None) is not None]
            if not feats:
                raise RuntimeError("No patch features available for molecular prediction.")
                
            feats_t = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(self.device)
            slide_embed, _ = self.abmil(feats_t)
            logits = self.head(slide_embed)
            
            msi_prob = round(float(torch.sigmoid(logits['msi']).item()), 4)
            ebv_prob = round(float(torch.sigmoid(logits['ebv']).item()), 4)
            
            lauren_softmax = torch.softmax(logits['lauren'], dim=-1).squeeze().tolist()
            lauren_probs = {
                "intestinal": round(lauren_softmax[0], 4),
                "diffuse": round(lauren_softmax[1], 4),
                "mixed": round(lauren_softmax[2], 4)
            }
            lauren_class = max(lauren_probs, key=lauren_probs.get)
            
            # HER2 prediction removed from H&E logic per clinical safety review.
            # HER2 status must be extracted via IHC/EHR reports in ClinicalContextAgent.
            her2_prob = 0.0

        msi_status = (
            MSIStatus.MSI_HIGH
            if msi_prob >= self.cfg.msi_threshold
            else MSIStatus.MSS
        )

        lauren_map = {
            "intestinal": LaurenClassification.INTESTINAL,
            "diffuse": LaurenClassification.DIFFUSE,
            "mixed": LaurenClassification.MIXED,
        }

        # HER2
        her2_status = HER2Status.UNKNOWN

        # EBV
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
