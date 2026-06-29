"""
Layer 6 — Feedback Capture and MDT Exporter.

FeedbackCapture: records pathologist disagreements with AI outputs
in a structured JSONL log for model improvement feedback loops.

MDTExporter: generates a concise multi-disciplinary team (MDT) summary
for verbal case presentation in tumour boards.

Architecture doc note:
  "Feedback capture: pathologist disagreement recorded for model
   improvement — the biggest gap in the field is clinician-AI
   collaboration design, not model accuracy."
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from hakim_ai.types import PathologyReport, PipelineResult
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer6.feedback")


# ---------------------------------------------------------------------------
# Feedback data structures
# ---------------------------------------------------------------------------

@dataclass
class PathologistFeedback:
    """Structured disagreement record from a reviewing pathologist."""
    patient_id: str
    run_id: Optional[str]
    timestamp: str
    pathologist_id: str
    agrees_with_diagnosis: bool
    corrected_diagnosis: Optional[str] = None
    agrees_with_lauren: bool = True
    corrected_lauren: Optional[str] = None
    agrees_with_msi: bool = True
    corrected_msi: Optional[str] = None
    explanation_quality: int = 3         # 1–5 Likert scale
    explanation_comments: Optional[str] = None
    overall_usefulness: int = 3          # 1–5 Likert scale
    free_text_comments: Optional[str] = None
    tags: List[str] = field(default_factory=list)   # e.g. ["rare_subtype", "poor_quality"]


class FeedbackCapture:
    """
    Captures and persists pathologist feedback.
    Output: JSONL file (one JSON object per line).
    """

    def __init__(self, db_path: str = "outputs/feedback.jsonl"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def record(self, feedback: PathologistFeedback) -> None:
        """Append a feedback record to the JSONL store."""
        record_dict = asdict(feedback)
        with open(self.db_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record_dict) + "\n")
        logger.info(
            "Feedback recorded for patient %s (agrees=%s, explanation=%d/5)",
            feedback.patient_id,
            feedback.agrees_with_diagnosis,
            feedback.explanation_quality,
        )

    def load_all(self) -> List[Dict[str, Any]]:
        """Load all feedback records."""
        if not os.path.exists(self.db_path):
            return []
        records = []
        with open(self.db_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def agreement_rate(self) -> float:
        """Compute overall pathologist-AI agreement rate."""
        records = self.load_all()
        if not records:
            return float("nan")
        agreements = sum(1 for r in records if r.get("agrees_with_diagnosis", False))
        return agreements / len(records)

    def from_dict(
        self,
        result: PipelineResult,
        pathologist_id: str,
        feedback_dict: Dict[str, Any],
    ) -> PathologistFeedback:
        """Construct a PathologistFeedback from a web-form / API payload."""
        return PathologistFeedback(
            patient_id=result.patient_id,
            run_id=result.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            pathologist_id=pathologist_id,
            **{k: v for k, v in feedback_dict.items()
               if k in PathologistFeedback.__dataclass_fields__
               and k not in ("patient_id", "run_id", "timestamp", "pathologist_id")},
        )


# ---------------------------------------------------------------------------
# MDT Exporter
# ---------------------------------------------------------------------------

class MDTExporter:
    """
    Generates an MDT (multi-disciplinary team) case summary.

    Output: structured text suitable for verbal presentation and
    import into presentation software.
    """

    def export(self, result: PipelineResult) -> str:
        """Return the MDT summary as a formatted text string."""
        if result.report is None:
            return f"[MDT SUMMARY — Patient {result.patient_id}]\nPipeline incomplete — no report generated."

        report = result.report
        diag = report.diagnosis
        mol = report.molecular_predictions
        expl = report.explanation
        conf = diag.overall_confidence

        lines = [
            "=" * 60,
            f"MDT CASE PRESENTATION — {report.patient_id}",
            f"Report generated: {report.timestamp}",
            "=" * 60,
            "",
            "DIAGNOSIS",
            "-" * 40,
            f"  Primary:    {diag.primary_diagnosis}",
            f"  Label:      {diag.diagnostic_label.value.upper()}",
            f"  WHO class:  {report.who_classification}",
            f"  Lauren:     {report.lauren_classification.value.capitalize()}",
            f"  Grade:      {diag.grade or 'Not determined'}",
            f"  TNM (est.): {diag.tnm_contribution or 'Requires resection staging'}",
            f"  AI Conf.:   {conf:.0%}",
            "",
            "BIOMARKER PREDICTIONS (H&E only — confirm with IHC/molecular testing)",
            "-" * 40,
            f"  MSI:   {mol.msi_status.value} (P={mol.msi_probability:.2f})",
            f"  HER2:  {mol.her2_status.value} (P={mol.her2_probability:.2f})",
            f"  EBV:   {mol.ebv_status.value} (P={mol.ebv_probability:.2f})",
            "",
            "KEY MORPHOLOGICAL FEATURES",
            "-" * 40,
        ]
        for feat in expl.key_morphological_features[:5]:
            lines.append(f"  • {feat}")

        if diag.differential_diagnoses:
            lines += ["", "DIFFERENTIAL DIAGNOSES", "-" * 40]
            for diff in diag.differential_diagnoses[:4]:
                lines.append(f"  • {diff}")

        if report.uncertainty_flags:
            lines += ["", "UNCERTAINTY FLAGS", "-" * 40]
            for flag in report.uncertainty_flags:
                lines.append(f"  ⚠ {flag}")

        lines += ["", "RECOMMENDATIONS FOR MDT", "-" * 40]
        for rec in report.recommendations:
            lines.append(f"  → {rec}")

        lines += [
            "",
            "EVIDENCE SUMMARY",
            "-" * 40,
            expl.narrative[:500] + "..." if len(expl.narrative) > 500 else expl.narrative,
            "",
            "=" * 60,
            "⚠  AI-assisted report. Pathologist sign-off required before clinical action.",
            "=" * 60,
        ]
        return "\n".join(lines)

    def save(self, result: PipelineResult, output_dir: str = "outputs") -> str:
        """Save MDT summary to a text file."""
        summary = self.export(result)
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"mdt_{result.patient_id}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(summary)
        logger.info("MDT summary saved to %s", path)
        return path