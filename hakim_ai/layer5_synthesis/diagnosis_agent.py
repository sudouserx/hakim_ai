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
        tnm = self._estimate_tnm_contribution(evidence, fusion)

        diagnostic_label = router_decision.label if router_decision else DiagnosticLabel.MALIGNANT
        
        # Override to benign if very low tumour fraction and no WHO match
        if evidence.segmentation.tumour_fraction < 0.05 and not verification.who_validation.matched_criteria:
            diagnostic_label = DiagnosticLabel.BENIGN
            primary = "Gastric mucosa, no evidence of malignancy"

        diagnosis = Diagnosis(
            primary_diagnosis=primary,
            diagnostic_label=diagnostic_label,
            differential_diagnoses=differentials,
            grade=grade,
            tnm_contribution=tnm,
            overall_confidence=verification.calibrated_confidence,
            supporting_findings=supporting,
        )

        logger.info("Diagnosis: %s (confidence=%.2f)", primary, diagnosis.overall_confidence)
        return diagnosis

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _grade_from_morphology(self, evidence: EvidenceBundle) -> Optional[str]:
        """Infer histological grade from patch description narratives or ML model."""
        narratives = " ".join(d.narrative.lower() for d in evidence.descriptions)
        if "poorly differentiated" in narratives or "poor differentiation" in narratives:
            return "poorly differentiated (Grade 3)"
        if "moderately differentiated" in narratives or "moderate differentiation" in narratives:
            return "moderately differentiated (Grade 2)"
        if "well differentiated" in narratives or "well-differentiated" in narratives:
            return "well differentiated (Grade 1)"
            
        if self.grade_model is not None and evidence.navigation.selected_patches:
            import numpy as np
            features = [p.feature_vector for p in evidence.navigation.selected_patches if getattr(p, "feature_vector", None) is not None]
            if features:
                mean_feat = np.mean(features, axis=0).tolist()
                return self.grade_model.predict_grade(mean_feat, self.device)
                
        return "indeterminate grade (IHC recommended)"

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

    def _estimate_tnm_contribution(
        self, evidence: EvidenceBundle, fusion: FusionResult
    ) -> Optional[str]:
        """
        Estimate T-stage contribution from H&E findings.
        Definitive staging requires surgical specimen with full assessment.
        """
        tumour_frac = evidence.segmentation.tumour_fraction
        if tumour_frac > 0.60:
            return "pT3-pT4 (estimated — confirm with full resection staging)"
        if tumour_frac > 0.30:
            return "pT2-pT3 (estimated — confirm with full resection staging)"
        return "pT1-pT2 (estimated from biopsy — full staging requires resection)"