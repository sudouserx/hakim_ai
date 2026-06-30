#!/usr/bin/env python3
"""
Command-line interface for the hakim_ai pipeline.

Usage examples
--------------
# Run with mock data (no real WSI needed):
python scripts/run_pipeline.py --patient-id P001 --wsi-path /path/to/slide.svs

# Run with clinical data:
python scripts/run_pipeline.py \\
    --patient-id P001 \\
    --wsi-path /path/to/slide.svs \\
    --age 62 --sex M \\
    --biopsy-location antrum \\
    --h-pylori \\
    --endoscopy "2.5cm ulcerative lesion in gastric antrum" \\
    --radiology /path/to/ct.dcm

# Use a custom config:
python scripts/run_pipeline.py --config config/kaggle.yaml --patient-id P001 --wsi-path slide.svs

# Save HTML report and MDT summary:
python scripts/run_pipeline.py --patient-id P001 --wsi-path slide.svs --save-report --save-mdt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Allow running from project root without installing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hakim_ai import HistopathologyPipeline, PipelineConfig
from hakim_ai.types import ClinicalInput, PipelineInput, WSIInput
from hakim_ai.utils import setup_logging


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="hakim_ai — Gastric cancer AI diagnostic pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core
    p.add_argument("--patient-id", required=True, help="Patient identifier")
    p.add_argument("--wsi-path", required=True, help="Path to whole-slide image (.svs/.ndpi/.tiff)")
    p.add_argument("--config", default=None, help="Path to YAML config file")
    p.add_argument("--run-id", default=None, help="Optional run ID for tracking")

    # Clinical data (optional)
    cg = p.add_argument_group("Clinical data (optional)")
    cg.add_argument("--age", type=int, default=None, help="Patient age in years")
    cg.add_argument("--sex", choices=["M", "F"], default=None)
    cg.add_argument("--biopsy-location", default=None, choices=["antrum", "body", "cardia", "fundus"])
    cg.add_argument("--h-pylori", action="store_true", default=False, help="H. pylori positive")
    cg.add_argument("--endoscopy", default=None, help="Free-text endoscopy findings")
    cg.add_argument("--ehr-notes", default=None, help="Free-text EHR clinical notes")
    cg.add_argument("--prior-treatment", action="append", default=[], dest="prior_treatments",
                    help="Prior treatments (repeat flag for multiple)")
    cg.add_argument("--molecular-report", default=None, help="Prior molecular test report text")
    cg.add_argument("--radiology", default=None, help="Path to CT/MRI file (optional)")

    # Output
    og = p.add_argument_group("Output options")
    og.add_argument("--output-dir", default="outputs", help="Directory for report files")
    og.add_argument("--save-report", action="store_true", help="Save HTML pathologist report")
    og.add_argument("--save-mdt", action="store_true", help="Save MDT text summary")
    og.add_argument("--json", action="store_true", dest="output_json", help="Print result as JSON")
    og.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    setup_logging(args.log_level)

    # Build config
    if args.config:
        cfg = PipelineConfig.from_yaml(args.config)
    else:
        cfg = PipelineConfig.default()
    cfg.ui.output_dir = args.output_dir
    cfg.log_level = args.log_level

    # Build pipeline input
    wsi_input = WSIInput(
        wsi_path=args.wsi_path,
        patient_id=args.patient_id,
    )

    clinical_input = None
    if any([args.age, args.sex, args.endoscopy, args.ehr_notes, args.h_pylori,
            args.biopsy_location, args.prior_treatments, args.molecular_report]):
        clinical_input = ClinicalInput(
            patient_id=args.patient_id,
            age=args.age,
            sex=args.sex,
            endoscopy_findings=args.endoscopy,
            ehr_notes=args.ehr_notes,
            h_pylori_status=args.h_pylori if args.h_pylori else None,
            biopsy_location=args.biopsy_location,
            prior_treatments=args.prior_treatments,
            molecular_reports=args.molecular_report,
        )

    pipeline_input = PipelineInput(
        wsi_input=wsi_input,
        clinical_input=clinical_input,
        radiology_path=args.radiology,
        run_id=args.run_id,
    )

    # Run pipeline
    pipeline = HistopathologyPipeline(cfg)
    print(f"\n🔬 Running hakim_ai pipeline for patient {args.patient_id}...")
    result = pipeline.run(pipeline_input)

    # Output
    if not result.is_successful():
        print(f"\n❌ Pipeline failed: {result.error}")
        if result.qc_result and not result.qc_result.passed:
            print(f"   QC rejection: {result.qc_result.rejection_reason}")
        return 1

    report = result.report
    print("\n" + "=" * 60)
    print(f"✅ Report generated in {result.pipeline_duration_seconds:.2f}s")
    print("=" * 60)
    print(f"\nPatient:    {report.patient_id}")
    print(f"Diagnosis:  {report.diagnosis.primary_diagnosis}")
    print(f"Label:      {report.diagnosis.diagnostic_label.value.upper()}")
    print(f"Confidence: {report.diagnosis.overall_confidence:.0%}")
    print(f"\nBiomarkers (H&E prediction):")
    for k, v in report.biomarker_summary.items():
        if k != "note":
            print(f"  {k}: {v}")

    if result.escalated_to_human:
        print("\n⚠  Case flagged for human review — see uncertainty flags.")

    if report.uncertainty_flags:
        print("\nUncertainty flags:")
        for f in report.uncertainty_flags:
            print(f"  ⚠ {f}")

    if args.save_report:
        path = pipeline.save_html_report(result)
        print(f"\n📄 HTML report saved to: {path}")

    if args.save_mdt:
        path = pipeline.save_mdt_summary(result)
        print(f"📋 MDT summary saved to: {path}")

    if args.output_json:
        print("\nJSON output:")
        print(json.dumps(result.to_dict(), indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())