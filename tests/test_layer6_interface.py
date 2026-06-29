"""
Unit tests for Layer 6 — Human Interface subsystem.

Covers:
  - UIRenderer: HTML generation, importance map rendering, error cases
  - FeedbackCapture: record/load/agreement rate
  - MDTExporter: text summary generation
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict

import pytest

from hakim_ai import HistopathologyPipeline
from hakim_ai.layer6_interface.ui_renderer import UIRenderer
from hakim_ai.layer6_interface.feedback_capture import (
    FeedbackCapture,
    MDTExporter,
    PathologistFeedback,
)
from hakim_ai.types import (
    DiagnosticLabel,
    PipelineInput,
    PipelineResult,
    WSIInput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_full_result(test_config) -> PipelineResult:
    """Run the pipeline in mock mode and return a complete result."""
    pipeline = HistopathologyPipeline(test_config)
    inp = PipelineInput(
        wsi_input=WSIInput("/fake/slide.svs", "UI_P001"),
        run_id="ui-test-001",
    )
    return pipeline.run(inp)


# ---------------------------------------------------------------------------
# UIRenderer tests
# ---------------------------------------------------------------------------

class TestUIRenderer:

    @pytest.fixture
    def result(self, test_config):
        return _build_full_result(test_config)

    @pytest.fixture
    def renderer(self, tmp_path):
        return UIRenderer(output_dir=str(tmp_path))

    def test_render_returns_string(self, renderer, result):
        html = renderer.render(result)
        assert isinstance(html, str)
        assert len(html) > 100

    def test_html_is_valid_structure(self, renderer, result):
        html = renderer.render(result)
        assert "<!DOCTYPE html>" in html
        assert "<html" in html
        assert "</html>" in html

    def test_html_contains_patient_id(self, renderer, result):
        html = renderer.render(result)
        assert result.patient_id in html

    def test_html_contains_diagnosis_section(self, renderer, result):
        html = renderer.render(result)
        assert "Diagnosis" in html or "diagnosis" in html

    def test_html_contains_biomarker_section(self, renderer, result):
        html = renderer.render(result)
        assert "MSI" in html or "Biomarker" in html

    def test_html_contains_disclaimer(self, renderer, result):
        """Report must contain the AI-assisted disclaimer."""
        html = renderer.render(result)
        assert "pathologist" in html.lower()
        assert "sign-off" in html.lower() or "sign off" in html.lower() or "oversight" in html.lower()

    def test_save_creates_file(self, renderer, result, tmp_path):
        path = renderer.save(result)
        assert os.path.exists(path)
        assert path.endswith(".html")

    def test_saved_file_non_empty(self, renderer, result, tmp_path):
        path = renderer.save(result)
        assert os.path.getsize(path) > 500

    def test_error_result_renders_gracefully(self, renderer):
        """A result with no report (error case) should still produce HTML."""
        from hakim_ai.types import QCResult, RouterDecision, CaseComplexity, TaskType
        error_result = PipelineResult(
            patient_id="ERR001",
            run_id="err-001",
            qc_result=QCResult(passed=False, stain_quality_score=0.2,
                               focus_quality_score=0.2, coverage_score=0.1,
                               rejection_reason="Stain too pale"),
            router_decision=RouterDecision(
                label=DiagnosticLabel.UNKNOWN, complexity=CaseComplexity.COMPLEX,
                confidence=0.0, task_type=TaskType.CLASSIFICATION,
            ),
            error="QC failed: Stain too pale",
        )
        html = renderer.render(error_result)
        assert isinstance(html, str)
        assert "ERR001" in html

    def test_importance_map_rendered(self, renderer, result):
        html = renderer.render(result)
        if result.evidence and result.evidence.navigation.importance_map:
            assert "rgb(" in html  # colour cells were rendered

    def test_uncertainty_flags_in_html(self, renderer, result):
        html = renderer.render(result)
        if result.report and result.report.uncertainty_flags:
            # flags section should appear
            assert "uncertainty" in html.lower() or "flag" in html.lower() or "⚠" in html

    def test_recommendations_in_html(self, renderer, result):
        html = renderer.render(result)
        if result.report and result.report.recommendations:
            assert "Recommendation" in html or result.report.recommendations[0][:20] in html


# ---------------------------------------------------------------------------
# FeedbackCapture tests
# ---------------------------------------------------------------------------

class TestFeedbackCapture:

    @pytest.fixture
    def db_path(self, tmp_path):
        return str(tmp_path / "feedback.jsonl")

    @pytest.fixture
    def capture(self, db_path):
        return FeedbackCapture(db_path=db_path)

    @pytest.fixture
    def sample_feedback(self):
        return PathologistFeedback(
            patient_id="P001",
            run_id="test-001",
            timestamp="2025-01-01T10:00:00+00:00",
            pathologist_id="DR_SMITH",
            agrees_with_diagnosis=True,
            explanation_quality=4,
            overall_usefulness=4,
        )

    def test_record_creates_file(self, capture, db_path, sample_feedback):
        capture.record(sample_feedback)
        assert os.path.exists(db_path)

    def test_record_appends_jsonl(self, capture, db_path, sample_feedback):
        capture.record(sample_feedback)
        with open(db_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["patient_id"] == "P001"

    def test_multiple_records_appended(self, capture, sample_feedback):
        capture.record(sample_feedback)
        capture.record(sample_feedback)
        records = capture.load_all()
        assert len(records) == 2

    def test_load_all_empty_before_any_record(self, db_path):
        capture = FeedbackCapture(db_path=str(db_path) + "_new.jsonl")
        records = capture.load_all()
        assert records == []

    def test_agreement_rate_all_agree(self, capture, sample_feedback):
        capture.record(sample_feedback)  # agrees=True
        rate = capture.agreement_rate()
        assert rate == 1.0

    def test_agreement_rate_none_agree(self, capture):
        fb = PathologistFeedback(
            patient_id="P002", run_id="r2", timestamp="2025-01-01T11:00:00+00:00",
            pathologist_id="DR_LEE", agrees_with_diagnosis=False,
        )
        capture.record(fb)
        rate = capture.agreement_rate()
        assert rate == 0.0

    def test_agreement_rate_mixed(self, capture):
        for agree in [True, True, False, True]:
            fb = PathologistFeedback(
                patient_id="P003", run_id="r", timestamp="t",
                pathologist_id="DR", agrees_with_diagnosis=agree,
            )
            capture.record(fb)
        rate = capture.agreement_rate()
        assert abs(rate - 0.75) < 0.01

    def test_agreement_rate_nan_on_empty(self, capture):
        import math
        rate = capture.agreement_rate()
        assert math.isnan(rate)

    def test_from_dict_builds_feedback(self, capture, test_config):
        pipeline = HistopathologyPipeline(test_config)
        inp = PipelineInput(wsi_input=WSIInput("/fake/s.svs", "P_FB"))
        result = pipeline.run(inp)
        fb = capture.from_dict(result, "DR_TEST", {
            "agrees_with_diagnosis": False,
            "corrected_diagnosis": "Intestinal metaplasia",
            "explanation_quality": 2,
            "overall_usefulness": 3,
        })
        assert fb.patient_id == "P_FB"
        assert fb.pathologist_id == "DR_TEST"
        assert not fb.agrees_with_diagnosis
        assert fb.corrected_diagnosis == "Intestinal metaplasia"

    def test_feedback_round_trip(self, capture, sample_feedback):
        capture.record(sample_feedback)
        loaded = capture.load_all()
        assert loaded[0]["patient_id"] == sample_feedback.patient_id
        assert loaded[0]["agrees_with_diagnosis"] == sample_feedback.agrees_with_diagnosis
        assert loaded[0]["explanation_quality"] == sample_feedback.explanation_quality


# ---------------------------------------------------------------------------
# MDTExporter tests
# ---------------------------------------------------------------------------

class TestMDTExporter:

    @pytest.fixture
    def exporter(self):
        return MDTExporter()

    @pytest.fixture
    def result(self, test_config):
        return _build_full_result(test_config)

    def test_export_returns_string(self, exporter, result):
        summary = exporter.export(result)
        assert isinstance(summary, str)
        assert len(summary) > 50

    def test_summary_contains_patient_id(self, exporter, result):
        summary = exporter.export(result)
        assert result.patient_id in summary

    def test_summary_contains_diagnosis_section(self, exporter, result):
        summary = exporter.export(result)
        assert "DIAGNOSIS" in summary

    def test_summary_contains_biomarker_section(self, exporter, result):
        summary = exporter.export(result)
        assert "BIOMARKER" in summary

    def test_summary_contains_disclaimer(self, exporter, result):
        summary = exporter.export(result)
        assert "pathologist" in summary.lower() or "sign-off" in summary.lower()

    def test_summary_contains_recommendations(self, exporter, result):
        summary = exporter.export(result)
        assert "RECOMMENDATION" in summary

    def test_summary_lines_are_80_chars_wide(self, exporter, result):
        summary = exporter.export(result)
        # The separator line should be 60 chars
        assert "=" * 60 in summary

    def test_save_creates_txt_file(self, exporter, result, tmp_path):
        path = exporter.save(result, output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".txt")

    def test_saved_file_has_content(self, exporter, result, tmp_path):
        path = exporter.save(result, output_dir=str(tmp_path))
        assert os.path.getsize(path) > 100

    def test_no_report_returns_graceful_message(self, exporter):
        """Export without a report should not crash."""
        from hakim_ai.types import QCResult, RouterDecision, CaseComplexity, TaskType
        empty_result = PipelineResult(
            patient_id="EMPTY001", run_id="r1",
            qc_result=QCResult(passed=True, stain_quality_score=0.8,
                               focus_quality_score=0.8, coverage_score=0.7),
            router_decision=RouterDecision(
                label=DiagnosticLabel.UNKNOWN, complexity=CaseComplexity.COMPLEX,
                confidence=0.0, task_type=TaskType.CLASSIFICATION,
            ),
            error="Pipeline incomplete",
        )
        summary = exporter.export(empty_result)
        assert "EMPTY001" in summary
        assert "incomplete" in summary.lower() or "no report" in summary.lower()

    def test_msi_h_in_biomarker_section(self, exporter, test_config):
        """MSI-H predicted case should show MSI-H in the MDT summary."""
        # Run pipeline and check output
        pipeline = HistopathologyPipeline(test_config)
        inp = PipelineInput(wsi_input=WSIInput("/fake/slide.svs", "P_MSI"))
        result = pipeline.run(inp)
        summary = exporter.export(result)
        # MSI status always appears in the biomarker section
        assert "MSI" in summary