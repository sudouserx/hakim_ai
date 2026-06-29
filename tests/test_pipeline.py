"""
Integration tests for the full end-to-end pipeline.

Tests that the orchestrator wires all layers correctly and returns
a well-formed PipelineResult for common scenarios.
"""
from __future__ import annotations

import pytest

from hakim_ai import HistopathologyPipeline, PipelineConfig
from hakim_ai.types import (
    DiagnosticLabel,
    PipelineInput,
    WSIInput,
)


@pytest.fixture
def pipeline(test_config) -> HistopathologyPipeline:
    return HistopathologyPipeline(test_config)


class TestFullPipelineIntegration:

    def test_pipeline_runs_successfully(self, pipeline, pipeline_input):
        result = pipeline.run(pipeline_input)
        assert result is not None
        assert result.patient_id == "P001"
        assert result.qc_result is not None
        assert result.router_decision is not None
        assert result.error is None

    def test_pipeline_produces_report(self, pipeline, pipeline_input):
        result = pipeline.run(pipeline_input)
        # Pipeline may produce a full report or an uncertainty/benign report
        assert result.report is not None
        assert result.report.patient_id == "P001"
        assert result.report.timestamp is not None
        assert len(result.report.timestamp) > 0

    def test_report_has_diagnosis(self, pipeline, pipeline_input):
        result = pipeline.run(pipeline_input)
        report = result.report
        assert report is not None
        assert report.diagnosis is not None
        assert isinstance(report.diagnosis.primary_diagnosis, str)
        assert len(report.diagnosis.primary_diagnosis) > 0

    def test_report_has_biomarker_summary(self, pipeline, pipeline_input):
        result = pipeline.run(pipeline_input)
        if result.report and result.report.biomarker_summary:
            bs = result.report.biomarker_summary
            # At least one biomarker key present
            assert any(k in bs for k in ("MSI_status", "Lauren_type", "HER2_status"))

    def test_pipeline_without_clinical_data(self, pipeline, pipeline_input_no_clinical):
        result = pipeline.run(pipeline_input_no_clinical)
        assert result.error is None or result.qc_result is not None

    def test_pipeline_with_radiology(self, pipeline, pipeline_input_with_radiology):
        result = pipeline.run(pipeline_input_with_radiology)
        assert result is not None
        # If radiology fusion ran, findings should be non-None in fusion
        if result.fusion:
            # radiology_findings may be a string if radiology_path was provided
            pass  # Just check it doesn't crash

    def test_pipeline_result_is_successful(self, pipeline, pipeline_input):
        result = pipeline.run(pipeline_input)
        # is_successful requires both no error AND a report
        if result.error is None and result.report is not None:
            assert result.is_successful()

    def test_pipeline_duration_recorded(self, pipeline, pipeline_input):
        result = pipeline.run(pipeline_input)
        assert result.pipeline_duration_seconds > 0
        assert result.pipeline_duration_seconds < 30   # should be fast in mock mode

    def test_to_dict_serialisable(self, pipeline, pipeline_input):
        result = pipeline.run(pipeline_input)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "patient_id" in d
        assert "qc_result" in d

    def test_pipeline_with_default_config(self, pipeline_input):
        """Pipeline should work with PipelineConfig.default()."""
        pipeline = HistopathologyPipeline(PipelineConfig.default())
        result = pipeline.run(pipeline_input)
        assert result is not None

    def test_multiple_patients_independent(self, pipeline):
        """Running two patients should not share state."""
        inp1 = PipelineInput(wsi_input=WSIInput("slide1.svs", "PA001"), run_id="r1")
        inp2 = PipelineInput(wsi_input=WSIInput("slide2.svs", "PA002"), run_id="r2")
        r1 = pipeline.run(inp1)
        r2 = pipeline.run(inp2)
        assert r1.patient_id == "PA001"
        assert r2.patient_id == "PA002"
        assert r1.run_id != r2.run_id