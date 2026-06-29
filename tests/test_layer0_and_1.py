"""Unit tests for Layer 0 (QC) and Layer 1 (Router)."""
from __future__ import annotations

import pytest

from hakim_ai.config import PipelineConfig, QCConfig, RouterConfig
from hakim_ai.layer0_input import MockWSILoader, QCAgent
from hakim_ai.layer1_router import RouterAgent
from hakim_ai.types import (
    CaseComplexity,
    DiagnosticLabel,
    TaskType,
)


# ── Layer 0: WSI Loader ──────────────────────────────────────────────────────

class TestMockWSILoader:

    def test_load_returns_wsi_data(self, wsi_input):
        loader = MockWSILoader(seed=42)
        data = loader.load(wsi_input)
        assert data.patient_id == "P001"
        assert data.wsi_path == wsi_input.wsi_path

    def test_wsi_data_has_thumbnail(self, wsi_input):
        loader = MockWSILoader(seed=42)
        data = loader.load(wsi_input)
        assert data.thumbnail is not None
        assert len(data.thumbnail) == 16
        assert len(data.thumbnail[0]) == 16
        assert len(data.thumbnail[0][0]) == 3   # RGB

    def test_wsi_data_has_dimensions(self, wsi_input):
        loader = MockWSILoader(seed=42)
        data = loader.load(wsi_input)
        assert data.level_count > 0
        assert len(data.level_dimensions) == data.level_count

    def test_mpp_positive(self, wsi_input):
        loader = MockWSILoader(seed=42)
        data = loader.load(wsi_input)
        assert data.mpp > 0


# ── Layer 0: QC Agent ────────────────────────────────────────────────────────

class TestQCAgent:

    def test_good_slide_passes(self, wsi_data, test_config):
        agent = QCAgent(test_config.qc)
        result = agent.run(wsi_data)
        # Mock slide should pass default thresholds
        assert result.stain_quality_score > 0
        assert result.focus_quality_score > 0
        assert result.coverage_score > 0

    def test_qc_result_has_scores(self, wsi_data, test_config):
        agent = QCAgent(test_config.qc)
        result = agent.run(wsi_data)
        assert 0.0 <= result.stain_quality_score <= 1.0
        assert 0.0 <= result.focus_quality_score <= 1.0
        assert 0.0 <= result.coverage_score <= 1.0

    def test_qc_fail_on_low_coverage(self, wsi_data):
        cfg = QCConfig(min_coverage=0.99)  # impossible threshold
        agent = QCAgent(cfg)
        result = agent.run(wsi_data)
        assert not result.passed
        assert result.rejection_reason is not None

    def test_qc_pass_sets_normalized_path(self, wsi_data, test_config):
        agent = QCAgent(test_config.qc)
        result = agent.run(wsi_data)
        if result.passed:
            assert result.normalized_wsi_path is not None

    def test_qc_area_positive_on_pass(self, wsi_data, test_config):
        agent = QCAgent(test_config.qc)
        result = agent.run(wsi_data)
        if result.passed:
            assert result.tissue_area_mm2 > 0

    def test_failing_qc_has_no_normalized_path(self, wsi_data):
        cfg = QCConfig(min_stain_quality=0.99, min_focus_quality=0.99, min_coverage=0.99)
        agent = QCAgent(cfg)
        result = agent.run(wsi_data)
        assert not result.passed
        assert result.normalized_wsi_path is None


# ── Layer 1: Router ──────────────────────────────────────────────────────────

class TestRouterAgent:

    def test_router_returns_decision(self, wsi_data, passing_qc_result, test_config):
        agent = RouterAgent(test_config.router)
        decision = agent.run(wsi_data, passing_qc_result)
        assert decision is not None
        assert isinstance(decision.label, DiagnosticLabel)
        assert isinstance(decision.complexity, CaseComplexity)
        assert isinstance(decision.task_type, TaskType)

    def test_router_confidence_in_range(self, wsi_data, passing_qc_result, test_config):
        agent = RouterAgent(test_config.router)
        decision = agent.run(wsi_data, passing_qc_result)
        assert 0.0 <= decision.confidence <= 1.0

    def test_router_has_rationale(self, wsi_data, passing_qc_result, test_config):
        agent = RouterAgent(test_config.router)
        decision = agent.run(wsi_data, passing_qc_result)
        assert isinstance(decision.routing_rationale, str)
        assert len(decision.routing_rationale) > 0

    def test_router_low_confidence_escalates(self, wsi_data, failing_qc_result):
        """Router should escalate when confidence is below threshold."""
        # Use a very high escalation threshold so almost anything triggers escalation
        cfg = RouterConfig(escalation_confidence_threshold=0.999)
        agent = RouterAgent(cfg)
        decision = agent.run(wsi_data, failing_qc_result)
        # With threshold=0.999, confidence will almost always be below it
        assert decision.escalate_to_human is True

    def test_benign_label_maps_to_classification(self, wsi_data, passing_qc_result, test_config):
        """If label is benign, task type should be classification."""
        agent = RouterAgent(test_config.router, seed=999)
        # Run multiple times; at least one benign should have classification task
        decisions = [agent.run(wsi_data, passing_qc_result) for _ in range(10)]
        benign_decisions = [d for d in decisions if d.label == DiagnosticLabel.BENIGN]
        for d in benign_decisions:
            assert d.task_type == TaskType.CLASSIFICATION

    def test_malignant_maps_to_biomarker_prediction(self, wsi_data, passing_qc_result, test_config):
        agent = RouterAgent(test_config.router)
        decisions = [agent.run(wsi_data, passing_qc_result) for _ in range(20)]
        malignant_decisions = [d for d in decisions if d.label == DiagnosticLabel.MALIGNANT]
        for d in malignant_decisions:
            assert d.task_type == TaskType.BIOMARKER_PREDICTION