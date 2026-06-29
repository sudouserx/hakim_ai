#!/usr/bin/env python3
"""
hakim_ai — Full-pipeline demonstration using synthetic data.

Demonstrates:
  1. Building a pipeline with default (mock) config
  2. Running a gastric cancer case with clinical context
  3. Running a second case with radiology data
  4. Demonstrating the abstention path (low-quality slide)
  5. Recording pathologist feedback
  6. Generating HTML + MDT outputs

No real WSI files, GPU, or model weights are required.
All agents run in mock mode — deterministic synthetic outputs.

Run:  python scripts/demo.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hakim_ai import HistopathologyPipeline, PipelineConfig
from hakim_ai.types import ClinicalInput, PipelineInput, WSIInput
from hakim_ai.layer6_interface.feedback_capture import (
    FeedbackCapture, MDTExporter, PathologistFeedback,
)
from hakim_ai.layer6_interface.ui_renderer import UIRenderer
from hakim_ai.utils import setup_logging

DIVIDER = "=" * 60


def section(title: str) -> None:
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


def run_demo() -> None:
    setup_logging("INFO")
    output_dir = tempfile.mkdtemp(prefix="histopath_ai_demo_")

    print(f"\n{'#' * 60}")
    print(f"  hakim_ai — Gastric Cancer AI Pipeline Demo")
    print(f"  All outputs will be written to: {output_dir}")
    print(f"{'#' * 60}")

    # ── Build pipeline ──────────────────────────────────────────────── #
    section("Step 1: Building pipeline (mock mode)")
    cfg = PipelineConfig.default()
    cfg.ui.output_dir = output_dir
    cfg.ui.html_report = True
    cfg.ui.mdt_export = True
    cfg.log_level = "WARNING"   # Quieter for demo output

    pipeline = HistopathologyPipeline(cfg)
    print("✅ Pipeline initialised with mock foundation models.")
    print("   (Set config.mock_mode=False + HF_TOKEN to load real weights)")

    # ── Case 1: Full clinical context ───────────────────────────────── #
    section("Step 2: Case 1 — MSI-suspected gastric adenocarcinoma")

    inp1 = PipelineInput(
        wsi_input=WSIInput(
            wsi_path="/data/slides/TCGA-BR-4253.svs",
            patient_id="TCGA-BR-4253",
            scanner_model="Aperio GT450 DX",
            magnification=40.0,
        ),
        clinical_input=ClinicalInput(
            patient_id="TCGA-BR-4253",
            age=67,
            sex="M",
            biopsy_location="antrum",
            h_pylori_status=True,
            endoscopy_findings=(
                "3.2cm irregular ulcerative lesion at the gastric antrum. "
                "Friable mucosa with raised edges. Biopsy taken from lesion edge."
            ),
            ehr_notes=(
                "67-year-old male presenting with 3-month history of epigastric pain, "
                "dysphagia, and 8kg weight loss. No prior cancer history. "
                "Family history: father had colorectal cancer aged 74."
            ),
            family_history="Father: colorectal cancer age 74",
        ),
        run_id="demo-case-1",
    )

    print(f"  Patient: {inp1.wsi_input.patient_id}")
    print(f"  Clinical: {inp1.clinical_input.age}y {inp1.clinical_input.sex}, "
          f"antrum biopsy, H. pylori +ve")

    result1 = pipeline.run(inp1)

    if result1.is_successful():
        r = result1.report
        print(f"\n  ✅ Report generated ({result1.pipeline_duration_seconds:.2f}s)")
        print(f"  Diagnosis:  {r.diagnosis.primary_diagnosis[:70]}")
        print(f"  Label:      {r.diagnosis.diagnostic_label.value.upper()}")
        print(f"  Confidence: {r.diagnosis.overall_confidence:.0%}")
        print(f"  Grade:      {r.diagnosis.grade or 'pending'}")
        print(f"\n  Biomarker predictions (H&E only):")
        for k, v in r.biomarker_summary.items():
            if k != "note":
                print(f"    {k:20s}: {v}")
        if r.uncertainty_flags:
            print(f"\n  Uncertainty flags:")
            for flag in r.uncertainty_flags:
                print(f"    ⚠ {flag}")
        print(f"\n  Top explanation excerpt:")
        excerpt = r.explanation.narrative[:300].replace("\n\n", "\n  ")
        print(f"  {excerpt}...")
        if r.explanation.counterfactual_note:
            print(f"\n  Counterfactual: {r.explanation.counterfactual_note[:120]}...")
    else:
        print(f"  ⚠ Pipeline completed with issue: {result1.error}")

    # ── Case 2: Radiology + pathology fusion ────────────────────────── #
    section("Step 3: Case 2 — Radiology-pathology multimodal fusion")

    inp2 = PipelineInput(
        wsi_input=WSIInput(
            wsi_path="/data/slides/TCGA-R1-A8MT.svs",
            patient_id="TCGA-R1-A8MT",
        ),
        clinical_input=ClinicalInput(
            patient_id="TCGA-R1-A8MT",
            age=54,
            sex="F",
            biopsy_location="body",
            h_pylori_status=False,
            molecular_reports="Prior HER2 IHC: 2+ (borderline). FISH pending.",
        ),
        radiology_path="/data/imaging/TCGA-R1-A8MT_ct.dcm",
        run_id="demo-case-2",
    )

    result2 = pipeline.run(inp2)

    if result2.is_successful():
        r = result2.report
        print(f"  ✅ Multimodal report for {inp2.wsi_input.patient_id}")
        print(f"  HER2 prediction: {r.biomarker_summary.get('HER2_status', 'N/A')}")
        print(f"  Lauren type:     {r.biomarker_summary.get('Lauren_type', 'N/A')}")
        if result2.fusion and result2.fusion.radiology_findings:
            print(f"  Radiology note:  {result2.fusion.radiology_findings[:100]}...")
    else:
        print(f"  ⚠ Case 2: {result2.error}")

    # ── Case 3: Low-quality slide → abstention ──────────────────────── #
    section("Step 4: Case 3 — Low-quality slide (abstention demonstration)")

    # Use a config with very high QC thresholds to force rejection / abstention
    strict_cfg = PipelineConfig.default()
    strict_cfg.qc.min_stain_quality = 0.95   # Near-impossible threshold
    strict_cfg.qc.min_focus_quality = 0.95
    strict_cfg.qc.min_coverage = 0.95
    strict_cfg.log_level = "WARNING"
    strict_pipeline = HistopathologyPipeline(strict_cfg)

    inp3 = PipelineInput(
        wsi_input=WSIInput(
            wsi_path="/data/slides/poor_quality_slide.svs",
            patient_id="POOR-QC-001",
        ),
        run_id="demo-case-3",
    )

    result3 = strict_pipeline.run(inp3)

    if not result3.qc_result.passed:
        print(f"  QC REJECTED: {result3.qc_result.rejection_reason}")
        print(f"  Stain score: {result3.qc_result.stain_quality_score:.2f} "
              f"(required: {strict_cfg.qc.min_stain_quality})")
        print("  ✅ Pipeline correctly stopped — returned quality issue report.")
    elif result3.escalated_to_human:
        print(f"  ABSTENTION: {result3.verification.abstention_reason if result3.verification else 'N/A'}")
        print("  ✅ Pipeline escalated to human — uncertainty report generated.")
    else:
        print(f"  (Mock quality scores passed threshold — case processed normally)")

    # ── Save outputs ─────────────────────────────────────────────────── #
    section("Step 5: Saving outputs")

    full_pipeline = HistopathologyPipeline(cfg)  # use original config with output_dir set

    if result1.is_successful():
        html_path = full_pipeline.save_html_report(result1)
        print(f"  HTML report: {html_path}")

        mdt_path = full_pipeline.save_mdt_summary(result1)
        print(f"  MDT summary: {mdt_path}")

        # Show a snippet of the MDT text
        with open(mdt_path) as f:
            mdt_lines = f.readlines()[:20]
        print("\n  MDT Summary Preview:")
        for line in mdt_lines:
            print(f"    {line}", end="")

    # ── Feedback capture ─────────────────────────────────────────────── #
    section("Step 6: Recording pathologist feedback")

    feedback_db = os.path.join(output_dir, "feedback.jsonl")
    capture = FeedbackCapture(db_path=feedback_db)

    # Pathologist agrees with first case
    fb1 = PathologistFeedback(
        patient_id=result1.patient_id,
        run_id=result1.run_id,
        timestamp="2025-06-01T09:30:00+00:00",
        pathologist_id="DR_GARCIA",
        agrees_with_diagnosis=True,
        explanation_quality=4,
        overall_usefulness=4,
        explanation_comments="TIL density rationale for MSI-H was helpful and accurate.",
        tags=["msi_h", "gastric_adenocarcinoma"],
    )
    capture.record(fb1)

    # Pathologist disagrees with second case's Lauren type
    if result2.is_successful():
        fb2 = PathologistFeedback(
            patient_id=result2.patient_id,
            run_id=result2.run_id,
            timestamp="2025-06-01T09:45:00+00:00",
            pathologist_id="DR_GARCIA",
            agrees_with_diagnosis=True,
            agrees_with_lauren=False,
            corrected_lauren="mixed",
            explanation_quality=3,
            overall_usefulness=4,
            free_text_comments="Lauren type is mixed, not purely intestinal. Both components are present.",
            tags=["lauren_correction", "mixed_type"],
        )
        capture.record(fb2)

    records = capture.load_all()
    rate = capture.agreement_rate()
    print(f"  Feedback records saved: {len(records)}")
    print(f"  Diagnostic agreement rate: {rate:.0%}")
    print(f"  Feedback DB: {feedback_db}")

    # ── Summary ──────────────────────────────────────────────────────── #
    section("Demo complete")
    print(f"  Outputs written to: {output_dir}")
    print(f"  Files generated:")
    for fname in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, fname)
        size = os.path.getsize(fpath)
        print(f"    {fname:<40s} ({size:,} bytes)")

    print(f"\n  Next steps to use real models:")
    print(f"    1. pip install 'hakim_ai[models]'")
    print(f"    2. export HF_TOKEN=<your_huggingface_token>")
    print(f"    3. Set mock_mode: false in config/default.yaml")
    print(f"    4. python scripts/run_pipeline.py --wsi-path real_slide.svs --patient-id P001")


if __name__ == "__main__":
    run_demo()