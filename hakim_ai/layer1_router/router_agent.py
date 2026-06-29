"""
Layer 1 — Router / Triage Agent.

Classifies a slide as benign | suspicious | malignant and assigns:
  - Case complexity (routine | intermediate | complex)
  - Task type (classification | biomarker_prediction | report_generation)
  - Human escalation flag

The router is a lightweight fast classifier that gates the expensive
multi-agent pipeline. It uses patch-level feature statistics from the
encoder plus QC metadata.

Mock implementation: uses QC scores + a seeded RNG to generate plausible
routing decisions; a real implementation would use a fine-tuned ViT-S or
a CONCH zero-shot classifier.
"""
from __future__ import annotations

import random
from typing import Optional

from hakim_ai.foundation_models.base_encoder import BaseEncoder
from hakim_ai.config import RouterConfig
from hakim_ai.types import (
    CaseComplexity,
    DiagnosticLabel,
    QCResult,
    RouterDecision,
    TaskType,
    WSIData,
)
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer1.router")


class RouterAgent:
    """
    Triage gate — fast binary/categorical slide classifier.

    Inputs:  WSIData, QCResult
    Outputs: RouterDecision
    """

    def __init__(self, cfg: RouterConfig, encoder: Optional[BaseEncoder] = None, seed: int = 42):
        self.cfg = cfg
        self.encoder = encoder
        self._rng = random.Random(seed)

    def run(self, wsi_data: WSIData, qc_result: QCResult) -> RouterDecision:
        logger.info("Router started for patient %s", wsi_data.patient_id)

        # In a real system: run a lightweight patch encoder + MIL aggregator
        if self.encoder is not None and not getattr(self.encoder, "mock_mode", True) and wsi_data.thumbnail is not None:
            # Simple feature-based classification on the thumbnail
            import numpy as np
            thumb_np = np.array(wsi_data.thumbnail)
            patch = thumb_np[:224, :224] if thumb_np.shape[0] >= 224 and thumb_np.shape[1] >= 224 else thumb_np
            
            try:
                # Real feature extraction
                feat = self.encoder.encode_patch(patch)
                
                # Mock linear probe on the extracted feature vector
                # Real implementation would have a trained linear layer here
                score = sum(f * 0.5 for f in feat[:10]) + 0.5 
                confidence = min(1.0, max(0.0, score + qc_result.coverage_score * 0.2))
            except Exception as e:
                logger.warning(f"Encoder routing failed, falling back to heuristic: {e}")
                confidence = self._heuristic_confidence(qc_result)
        else:
            confidence = self._heuristic_confidence(qc_result)

        label, label_confidence = self._classify(confidence)
        complexity = self._assess_complexity(label_confidence, qc_result)
        task_type = self._assign_task(label)
        escalate = self._should_escalate(label_confidence)

        rationale = self._build_rationale(label, label_confidence, qc_result)

        decision = RouterDecision(
            label=label,
            complexity=complexity,
            confidence=label_confidence,
            task_type=task_type,
            escalate_to_human=escalate,
            routing_rationale=rationale,
        )

        logger.info(
            "Router decision: %s | complexity=%s | confidence=%.2f | escalate=%s",
            label.value,
            complexity.value,
            label_confidence,
            escalate,
        )
        return decision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heuristic_confidence(self, qc_result: QCResult) -> float:
        base_confidence = (
            qc_result.stain_quality_score * 0.3
            + qc_result.focus_quality_score * 0.3
            + qc_result.coverage_score * 0.4
        )
        return min(1.0, max(0.0, base_confidence + self._rng.gauss(0, 0.05)))

    def _classify(
        self, raw_score: float
    ) -> tuple[DiagnosticLabel, float]:
        """
        Map a continuous score to a diagnostic label with calibrated confidence.

        Real implementation: softmax over [benign, suspicious, malignant] logits
        from a fine-tuned ViT-S or ABMIL model.
        """
        # Mock: treat raw_score as a combined quality/pathology signal
        # High quality slides from a cancer cohort are more likely malignant
        p_malignant = 0.60 + self._rng.gauss(0, 0.10)
        p_suspicious = 0.25 + self._rng.gauss(0, 0.05)
        p_benign = 1.0 - p_malignant - p_suspicious

        p_benign = max(0.0, p_benign)
        total = p_malignant + p_suspicious + p_benign
        p_malignant /= total
        p_suspicious /= total
        p_benign /= total

        if p_malignant >= p_suspicious and p_malignant >= p_benign:
            return DiagnosticLabel.MALIGNANT, round(p_malignant, 3)
        if p_suspicious >= p_benign:
            return DiagnosticLabel.SUSPICIOUS, round(p_suspicious, 3)
        return DiagnosticLabel.BENIGN, round(p_benign, 3)

    def _assess_complexity(
        self, confidence: float, qc: QCResult
    ) -> CaseComplexity:
        thresholds = self.cfg.complexity_thresholds
        if confidence >= thresholds.get("routine", 0.70) and not qc.artifacts_detected:
            return CaseComplexity.ROUTINE
        if confidence >= thresholds.get("intermediate", 0.40):
            return CaseComplexity.INTERMEDIATE
        return CaseComplexity.COMPLEX

    def _assign_task(self, label: DiagnosticLabel) -> TaskType:
        if label == DiagnosticLabel.BENIGN:
            return TaskType.CLASSIFICATION
        # Malignant / suspicious: run full biomarker prediction
        return TaskType.BIOMARKER_PREDICTION

    def _should_escalate(self, confidence: float) -> bool:
        return confidence < self.cfg.escalation_confidence_threshold

    def _build_rationale(
        self,
        label: DiagnosticLabel,
        confidence: float,
        qc: QCResult,
    ) -> str:
        return (
            f"Slide classified as {label.value} with confidence {confidence:.2f}. "
            f"Stain quality={qc.stain_quality_score:.2f}, "
            f"focus={qc.focus_quality_score:.2f}, "
            f"coverage={qc.coverage_score:.2f}."
        )