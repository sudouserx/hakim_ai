"""
Layer 5 — Report Agent.

Produces the final structured pathology report in WHO-compliant format.
Generates both a machine-readable dict and a human-readable text report.

The report includes:
  - WHO/Lauren classification
  - Biomarker predictions (MSI, HER2, EBV) with confidence
  - Histological grade
  - TNM contribution (from biopsy — full staging requires resection)
  - Uncertainty flags
  - Treatment pathway recommendations (based on NCCN guidelines)
  - Explicit MDT presentation summary
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hakim_ai.types import (
    Diagnosis,
    EvidenceBundle,
    Explanation,
    FusionResult,
    HER2Status,
    LaurenClassification,
    MSIStatus,
    PathologyReport,
    PipelineInput,
    RouterDecision,
    VerificationResult,
    QCResult,
)
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer5.report")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReportAgent:
    """
    Structured pathology report generator.

    Inputs:  PipelineInput, Diagnosis, Explanation, FusionResult, VerificationResult
    Outputs: PathologyReport
    """

    def run(
        self,
        pipeline_input: PipelineInput,
        diagnosis: Diagnosis,
        explanation: Explanation,
        fusion: FusionResult,
        verification: VerificationResult,
    ) -> PathologyReport:
        patient_id = pipeline_input.wsi_input.patient_id
        logger.info("Report generation started for patient %s", patient_id)

        mol = fusion.molecular
        lauren = mol.lauren_class
        who_class = (
            verification.who_validation.suggested_classification
            or "Gastric adenocarcinoma NOS"
        )

        biomarker_summary = self._build_biomarker_summary(mol)
        uncertainty_flags = self._collect_uncertainty_flags(verification, mol)
        recommendations = self._generate_recommendations(mol, diagnosis, verification)
        structured_fields = self._build_structured_fields(
            pipeline_input, diagnosis, mol, verification
        )

        report = PathologyReport(
            patient_id=patient_id,
            timestamp=_utc_now(),
            diagnosis=diagnosis,
            molecular_predictions=mol,
            explanation=explanation,
            lauren_classification=lauren,
            who_classification=who_class,
            uncertainty_flags=uncertainty_flags,
            recommendations=recommendations,
            structured_fields=structured_fields,
            biomarker_summary=biomarker_summary,
            report_version="1.0",
        )

        logger.info(
            "Report generated: %s | WHO=%s | flags=%d",
            diagnosis.primary_diagnosis[:60],
            who_class[:40],
            len(uncertainty_flags),
        )
        return report

    def generate_benign_report(
        self,
        pipeline_input: PipelineInput,
        qc_result: QCResult,
        router_decision: RouterDecision,
    ) -> PathologyReport:
        """Abbreviated report for cases classified as benign by the router."""
        from hakim_ai.types import (
            Diagnosis, DiagnosticLabel, Explanation, MolecularPrediction,
            EBVStatus, HER2Status, MSIStatus, LaurenClassification,
        )
        patient_id = pipeline_input.wsi_input.patient_id
        dummy_diagnosis = Diagnosis(
            primary_diagnosis="No evidence of malignancy on current biopsy",
            diagnostic_label=DiagnosticLabel.BENIGN,
            differential_diagnoses=["Reactive/inflammatory changes", "Intestinal metaplasia"],
            overall_confidence=router_decision.confidence,
        )
        dummy_mol = MolecularPrediction(
            msi_status=MSIStatus.UNKNOWN, msi_probability=0.0,
            lauren_class=LaurenClassification.UNKNOWN, lauren_confidence=0.0,
            her2_status=HER2Status.UNKNOWN, her2_probability=0.0,
            ebv_status=EBVStatus.UNKNOWN, ebv_probability=0.0,
        )
        dummy_explanation = Explanation(
            narrative=(
                f"Triage classifier assigned benign label with confidence "
                f"{router_decision.confidence:.2f}. Full multi-agent analysis not performed. "
                f"If clinical suspicion remains, request full pathology review."
            ),
            uncertainty_statement="Benign classification based on rapid triage only.",
        )
        return PathologyReport(
            patient_id=patient_id,
            timestamp=_utc_now(),
            diagnosis=dummy_diagnosis,
            molecular_predictions=dummy_mol,
            explanation=dummy_explanation,
            lauren_classification=LaurenClassification.UNKNOWN,
            who_classification="Benign gastric mucosa (triage classification)",
            uncertainty_flags=["Benign triage only — full analysis not performed"],
            recommendations=["Correlate with clinical findings", "Re-biopsy if suspicion persists"],
            report_version="1.0",
        )

    def generate_uncertainty_report(
        self,
        pipeline_input: PipelineInput,
        evidence: Optional[EvidenceBundle],
        fusion: Optional[FusionResult],
        verification: VerificationResult,
    ) -> PathologyReport:
        """Report for abstention cases — escalation summary."""
        from hakim_ai.types import (
            Diagnosis, DiagnosticLabel, Explanation, MolecularPrediction,
            EBVStatus, HER2Status, MSIStatus, LaurenClassification,
        )
        patient_id = pipeline_input.wsi_input.patient_id
        reason = verification.abstention_reason or "Low confidence — human review required"
        diagnosis = Diagnosis(
            primary_diagnosis="INCONCLUSIVE — Human review required",
            diagnostic_label=DiagnosticLabel.UNKNOWN,
            overall_confidence=verification.calibrated_confidence,
            supporting_findings=[reason],
        )
        mol = (
            fusion.molecular
            if fusion
            else MolecularPrediction(
                msi_status=MSIStatus.UNKNOWN, msi_probability=0.0,
                lauren_class=LaurenClassification.UNKNOWN, lauren_confidence=0.0,
                her2_status=HER2Status.UNKNOWN, her2_probability=0.0,
                ebv_status=EBVStatus.UNKNOWN, ebv_probability=0.0,
            )
        )
        explanation = Explanation(
            narrative=(
                f"The AI system has abstained from providing a definitive diagnosis.\n\n"
                f"Reason: {reason}\n\n"
                f"Calibrated confidence: {verification.calibrated_confidence:.2f} "
                f"(threshold: {0.35})\n\n"
                + (f"Consistency issues: {'; '.join(verification.consistency_issues)}" if verification.consistency_issues else "")
            ),
            uncertainty_statement=reason,
        )
        return PathologyReport(
            patient_id=patient_id,
            timestamp=_utc_now(),
            diagnosis=diagnosis,
            molecular_predictions=mol,
            explanation=explanation,
            lauren_classification=LaurenClassification.UNKNOWN,
            who_classification="INCONCLUSIVE",
            uncertainty_flags=[reason] + verification.consistency_issues,
            recommendations=[
                "Senior pathologist review required",
                "Additional IHC panel recommended",
                "Clinical correlation essential",
            ],
            report_version="1.0",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_biomarker_summary(self, mol) -> Dict[str, str]:
        return {
            "MSI_status": f"{mol.msi_status.value} (P={mol.msi_probability:.2f})",
            "Lauren_type": f"{mol.lauren_class.value} (conf={mol.lauren_confidence:.2f})",
            "HER2_status": f"{mol.her2_status.value} (P={mol.her2_probability:.2f})",
            "EBV_status": f"{mol.ebv_status.value} (P={mol.ebv_probability:.2f})",
            "note": "All predictions from H&E only; IHC/molecular testing required for clinical decision-making.",
        }

    def _collect_uncertainty_flags(self, verification, mol) -> List[str]:
        flags: List[str] = []
        if verification.is_ood:
            flags.append(f"OOD: {verification.uncertainty_source}")
        flags.extend(verification.consistency_issues)
        if mol.her2_status == HER2Status.EQUIVOCAL:
            flags.append("HER2 equivocal — FISH required")
        if mol.lauren_confidence < 0.60:
            flags.append(f"Lauren classification uncertain (conf={mol.lauren_confidence:.2f})")
        if mol.msi_probability > 0.4 and mol.msi_probability < 0.6:
            flags.append("MSI borderline — MMR IHC required")
        return flags

    def _generate_recommendations(self, mol, diagnosis, verification) -> List[str]:
        recs: List[str] = [
            "Pathologist review and sign-off required before clinical use.",
        ]
        if mol.msi_status == MSIStatus.MSI_HIGH:
            recs += [
                "MSI-H predicted: order MMR IHC (MLH1/MSH2/MSH6/PMS2).",
                "If confirmed MSI-H: consider pembrolizumab eligibility assessment.",
            ]
        if mol.her2_status in (HER2Status.POSITIVE, HER2Status.EQUIVOCAL):
            recs.append("HER2 prediction: order HER2 IHC (Hercep Test); ISH if IHC 2+.")
        if mol.ebv_probability > 0.40:
            recs.append("EBV probability elevated: consider EBER-ISH in situ hybridisation.")
        recs.append("Present findings at MDT with oncology, surgery, and radiology.")
        if verification.consistency_issues:
            recs.append("Consistency issues flagged — additional IHC panel recommended.")
        return recs

    def _build_structured_fields(
        self, pipeline_input, diagnosis, mol, verification
    ) -> Dict[str, Any]:
        return {
            "patient_id": pipeline_input.wsi_input.patient_id,
            "wsi_path": pipeline_input.wsi_input.wsi_path,
            "primary_diagnosis": diagnosis.primary_diagnosis,
            "diagnostic_label": diagnosis.diagnostic_label.value,
            "grade": diagnosis.grade,
            "tnm_contribution": diagnosis.tnm_contribution,
            "lauren_type": mol.lauren_class.value,
            "msi_status": mol.msi_status.value,
            "msi_probability": mol.msi_probability,
            "her2_status": mol.her2_status.value,
            "ebv_probability": mol.ebv_probability,
            "calibrated_confidence": verification.calibrated_confidence,
            "abstained": verification.abstain,
            "ood_detected": verification.is_ood,
            "who_classification": verification.who_validation.suggested_classification,
            "run_id": pipeline_input.run_id,
        }

    def to_json(self, report: PathologyReport) -> str:
        """Serialise report to JSON string."""
        import dataclasses
        return json.dumps(dataclasses.asdict(report), indent=2, default=str)