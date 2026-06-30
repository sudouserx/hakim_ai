"""
Layer 5 — Diagnosis Agent.

Synthesises all upstream evidence into a holistic final diagnosis:
  - Primary diagnosis string
  - Diagnostic label (malignant / suspicious / benign)
  - Differential diagnoses
  - Histological grade
  - Per-biomarker confidence

Design principle: the Diagnosis Agent integrates — it does not re-run
any image analysis. All inputs arrive as structured data from upstream
agents. This enables independent unit-testing of the synthesis logic.
"""
from __future__ import annotations

from typing import List, Optional

from hakim_ai.types import (
    Diagnosis,
    DiagnosticLabel,
    EvidenceBundle,
    FusionResult,
    HER2Status,
    LaurenClassification,
    MSIStatus,
    RouterDecision,
    VerificationResult,
)
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer5.diagnosis")





def _build_primary_diagnosis(
    fusion: FusionResult,
    who_suggested: Optional[str],
) -> str:
    mol = fusion.molecular
    lauren_str = mol.lauren_class.value.capitalize() if mol.lauren_class != LaurenClassification.UNKNOWN else ""
    who_term = who_suggested or "gastric adenocarcinoma NOS"

    parts = ["Gastric adenocarcinoma"]
    if lauren_str and lauren_str.lower() not in who_term.lower():
        parts.append(f"({lauren_str} type, Lauren classification)")
    parts.append(f"— {who_term}")
    return ", ".join(parts[:1]) + " " + " ".join(parts[1:])


def _build_differentials(
    fusion: FusionResult,
    evidence: EvidenceBundle,
) -> List[str]:
    diffs: List[str] = []
    mol = fusion.molecular
    narratives = " ".join(d.narrative.lower() for d in evidence.descriptions)

    # MSI-H opens immunotherapy pathway — worth listing
    if mol.msi_status == MSIStatus.MSI_HIGH:
        diffs.append("MSI-H / dMMR gastric adenocarcinoma (pembrolizumab eligible)")
    if mol.ebv_probability > 0.40:
        diffs.append("EBV-associated gastric carcinoma (lymphoepithelioma-like)")
    if mol.her2_status == HER2Status.POSITIVE:
        diffs.append("HER2-positive gastric adenocarcinoma (trastuzumab eligible)")
    if "signet" in narratives or mol.lauren_class == LaurenClassification.DIFFUSE:
        diffs.append("Poorly cohesive carcinoma with signet-ring cell morphology")
    if not diffs:
        diffs.append("High-grade gastric dysplasia (less likely given invasion pattern)")
    return diffs


class DiagnosisAgent:
    """
    Holistic diagnosis synthesis agent.

    Inputs:  EvidenceBundle, FusionResult, VerificationResult
    Outputs: Diagnosis
    """

    def __init__(self, use_gpu: bool = True):
        try:
            import torch
            from hakim_ai.models.grade_classifier import HistologicalGradeClassifier
            self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
            self.grade_model = HistologicalGradeClassifier.load_model("checkpoints/grade_classifier.pt", device=self.device)
        except ImportError:
            self.grade_model = None
            self.device = None

    # Safe abstention string for T-staging.  T-stage is determined by
    # invasion depth through anatomical layers (mucosa → serosa), which
    # cannot be inferred from 2D tumour area fraction on H&E patches.
    _TNM_ABSTENTION: str = (
        "Abstained: T-stage cannot be determined from 2D tumour area "
        "fraction. Requires 3D spatial evaluation or full surgical resection."
    )

    def run(
        self,
        evidence: EvidenceBundle,
        fusion: FusionResult,
        verification: VerificationResult,
        router_decision: Optional[RouterDecision] = None,
    ) -> Diagnosis:
        logger.info("Diagnosis synthesis started")

        grade = self._grade_from_morphology(evidence)
        who_suggested = verification.who_validation.suggested_classification
        primary = _build_primary_diagnosis(fusion, who_suggested)
        differentials = _build_differentials(fusion, evidence)
        supporting = self._collect_supporting_findings(evidence, fusion)

        diagnostic_label = router_decision.label if router_decision else DiagnosticLabel.MALIGNANT

        # Override to UNKNOWN/ESCALATE if very low tumour fraction and no WHO match
        if evidence.segmentation.tumour_fraction < 0.05 and not verification.who_validation.matched_criteria:
            diagnostic_label = DiagnosticLabel.UNKNOWN
            primary = "Indeterminate findings (low tumour fraction, no WHO criteria match) — manual review required"

        diagnosis = Diagnosis(
            primary_diagnosis=primary,
            diagnostic_label=diagnostic_label,
            differential_diagnoses=differentials,
            grade=grade,
            tnm_contribution=self._TNM_ABSTENTION,
            overall_confidence=verification.calibrated_confidence,
            supporting_findings=supporting,
        )

        logger.info("Diagnosis: %s (confidence=%.2f)", primary, diagnosis.overall_confidence)
        return diagnosis

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _grade_from_morphology(self, evidence: EvidenceBundle) -> Optional[str]:
        """Determine histological grade using ML feature extraction.

        Priority hierarchy:
          1. PyTorch ``HistologicalGradeClassifier`` over extracted patch
             features — this is the ground-truth source when available.
          2. Safe abstention — if the ML model is unavailable, we do NOT
             attempt naive substring matching on VLM narratives because
             simple ``in`` checks are vulnerable to negation errors
             (e.g. "NOT poorly differentiated" would incorrectly match
             "poorly differentiated").  Proper NLP semantic extraction
             or multimodal ML features are required to safely parse
             free-text narratives.
        """
        # --- Priority 1: ML model inference ---
        if self.grade_model is not None and evidence.navigation.selected_patches:
            import numpy as np
            features: list = [
                p.feature_vector
                for p in evidence.navigation.selected_patches
                if getattr(p, "feature_vector", None) is not None
            ]
            if features:
                mean_feat = np.mean(features, axis=0).tolist()
                grade = self.grade_model.predict_grade(mean_feat, self.device)
                logger.info("Grade determined by ML model: %s", grade)
                return grade

        # --- Priority 2: Safe abstention ---
        # NOTE: Do NOT fall back to substring matching on VLM narratives.
        # Naive ``if 'poorly differentiated' in text`` is fragile against
        # negation, hedging, and multi-clause sentences.  A production
        # replacement should use either:
        #   - NLP-based semantic role labelling / negation detection, or
        #   - Multimodal feature vectors fed to a trained classifier.
        logger.warning(
            "Grade model unavailable or no patch features — returning indeterminate grade"
        )
        return "indeterminate grade (requires ML feature extraction or pathologist review)"

    def _collect_supporting_findings(
        self, evidence: EvidenceBundle, fusion: FusionResult
    ) -> List[str]:
        findings: List[str] = []
        seg = evidence.segmentation
        mol = fusion.molecular

        if seg.tumour_fraction > 0.4:
            findings.append(f"Tumour fraction: {seg.tumour_fraction:.0%} of evaluated tissue")
        if seg.til_density > 0.15:
            findings.append(f"Elevated TIL density: {seg.til_density:.0%} (supports {mol.msi_status.value})")
        if seg.necrosis_fraction > 0.05:
            findings.append(f"Focal necrosis: {seg.necrosis_fraction:.0%}")
        if seg.stroma_fraction > 0.3:
            findings.append(f"Prominent desmoplastic stroma: {seg.stroma_fraction:.0%}")

        # Top morphological features from descriptions
        all_features: List[str] = []
        for desc in evidence.descriptions[:3]:
            all_features.extend(desc.morphological_features[:2])
        seen: set = set()
        for f in all_features:
            if f not in seen:
                findings.append(f"Morphological feature: {f}")
                seen.add(f)

        return findings[:8]

    # NOTE: _estimate_tnm_contribution has been intentionally removed.
    # T-stage is determined by invasion depth through anatomical wall layers
    # (mucosa, submucosa, muscularis propria, serosa), NOT by 2D tumour
    # area fraction.  The previous area-based heuristic was a clinical
    # safety hazard.  See _TNM_ABSTENTION for the safe replacement.