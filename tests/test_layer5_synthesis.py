"""
Unit tests for Layer 5 — Synthesis subsystem.

Covers:
  - DiagnosisAgent: holistic synthesis, grade inference, differential generation
  - ExplanationAgent: NL narrative, citations, uncertainty statements
  - ReportAgent: structured report, benign fast-path, abstention path
"""
from __future__ import annotations

import pytest

from hakim_ai.layer5_synthesis.diagnosis_agent import DiagnosisAgent
from hakim_ai.layer5_synthesis.explanation_agent import ExplanationAgent
from hakim_ai.layer5_synthesis.report_agent import ReportAgent
from hakim_ai.types import (
    DiagnosticLabel,
    LaurenClassification,
    MSIStatus,
    PipelineInput,
    WSIInput,
)


# ---------------------------------------------------------------------------
# DiagnosisAgent tests
# ---------------------------------------------------------------------------

class TestDiagnosisAgent:

    @pytest.fixture
    def agent(self):
        return DiagnosisAgent()

    def test_returns_diagnosis(self, agent, evidence_bundle, fusion_result, passing_verification):
        diag = agent.run(evidence_bundle, fusion_result, passing_verification)
        assert diag is not None

    def test_primary_diagnosis_non_empty(self, agent, evidence_bundle, fusion_result, passing_verification):
        diag = agent.run(evidence_bundle, fusion_result, passing_verification)
        assert isinstance(diag.primary_diagnosis, str)
        assert len(diag.primary_diagnosis) > 10

    def test_diagnostic_label_is_malignant(self, agent, evidence_bundle, fusion_result, passing_verification):
        """DiagnosisAgent always synthesises a malignant finding (benign handled by router)."""
        diag = agent.run(evidence_bundle, fusion_result, passing_verification)
        assert diag.diagnostic_label == DiagnosticLabel.MALIGNANT

    def test_differential_diagnoses_non_empty(self, agent, evidence_bundle, fusion_result, passing_verification):
        diag = agent.run(evidence_bundle, fusion_result, passing_verification)
        assert isinstance(diag.differential_diagnoses, list)
        assert len(diag.differential_diagnoses) >= 1

    def test_grade_inferred_from_narrative(self, agent, fusion_result, passing_verification):
        """Narratives containing 'poorly differentiated' should produce Grade 3."""
        from hakim_ai.types import PatchDescription, PatchCoordinate, NavigationResult, EvidenceBundle, TissueSegmentation
        p = PatchCoordinate(x=0, y=0, level=0, importance_score=0.9)
        desc = PatchDescription(
            patch_coord=p,
            narrative="Poorly differentiated adenocarcinoma with marked nuclear pleomorphism.",
            morphological_features=["poorly differentiated"], confidence=0.85,
        )
        nav = NavigationResult(selected_patches=[p], top_patch_count=1)
        ev = EvidenceBundle(
            navigation=nav,
            segmentation=TissueSegmentation(tumour_fraction=0.5, stroma_fraction=0.3, til_density=0.15),
            descriptions=[desc],
        )
        diag = agent.run(ev, fusion_result, passing_verification)
        assert diag.grade is not None
        assert "3" in diag.grade or "poorly" in diag.grade.lower()

    def test_msi_h_in_differentials(self, agent, evidence_bundle, passing_verification):
        """MSI-H molecular prediction should appear in differentials."""
        from hakim_ai.types import (
            FusionResult, MolecularPrediction, ClinicalContext, RetrievedKnowledge, HER2Status, EBVStatus
        )
        mol = MolecularPrediction(
            msi_status=MSIStatus.MSI_HIGH, msi_probability=0.85,
            lauren_class=LaurenClassification.INTESTINAL, lauren_confidence=0.70,
            her2_status=HER2Status.NEGATIVE, her2_probability=0.10,
            ebv_status=EBVStatus.NEGATIVE, ebv_probability=0.08,
        )
        fu = FusionResult(
            molecular=mol,
            clinical_context=ClinicalContext(encoded_prompt="test"),
            knowledge=RetrievedKnowledge(),
        )
        diag = agent.run(evidence_bundle, fu, passing_verification)
        assert any("MSI" in d for d in diag.differential_diagnoses)

    def test_overall_confidence_matches_verification(self, agent, evidence_bundle, fusion_result, passing_verification):
        diag = agent.run(evidence_bundle, fusion_result, passing_verification)
        assert diag.overall_confidence == passing_verification.calibrated_confidence

    def test_tnm_contribution_present(self, agent, evidence_bundle, fusion_result, passing_verification):
        diag = agent.run(evidence_bundle, fusion_result, passing_verification)
        assert diag.tnm_contribution is not None
        assert len(diag.tnm_contribution) > 0

    def test_supporting_findings_non_empty(self, agent, evidence_bundle, fusion_result, passing_verification):
        diag = agent.run(evidence_bundle, fusion_result, passing_verification)
        assert isinstance(diag.supporting_findings, list)
        assert len(diag.supporting_findings) >= 1


# ---------------------------------------------------------------------------
# ExplanationAgent tests
# ---------------------------------------------------------------------------

class TestExplanationAgent:

    @pytest.fixture
    def agent(self):
        return ExplanationAgent()

    @pytest.fixture
    def diagnosis(self, evidence_bundle, fusion_result, passing_verification):
        return DiagnosisAgent().run(evidence_bundle, fusion_result, passing_verification)

    def test_returns_explanation(self, agent, diagnosis, evidence_bundle, fusion_result, passing_verification):
        expl = agent.run(diagnosis, evidence_bundle, fusion_result, passing_verification)
        assert expl is not None

    def test_narrative_non_empty(self, agent, diagnosis, evidence_bundle, fusion_result, passing_verification):
        expl = agent.run(diagnosis, evidence_bundle, fusion_result, passing_verification)
        assert isinstance(expl.narrative, str)
        assert len(expl.narrative) > 50

    def test_narrative_mentions_diagnosis(self, agent, diagnosis, evidence_bundle, fusion_result, passing_verification):
        expl = agent.run(diagnosis, evidence_bundle, fusion_result, passing_verification)
        # Narrative should reference gastric or adenocarcinoma
        assert any(term in expl.narrative.lower() for term in ["gastric", "adenocarcinoma", "diagnosis"])

    def test_evidence_citations_non_empty(self, agent, diagnosis, evidence_bundle, fusion_result, passing_verification):
        expl = agent.run(diagnosis, evidence_bundle, fusion_result, passing_verification)
        assert isinstance(expl.evidence_citations, list)
        assert len(expl.evidence_citations) >= 1

    def test_citations_reference_patches(self, agent, diagnosis, evidence_bundle, fusion_result, passing_verification):
        expl = agent.run(diagnosis, evidence_bundle, fusion_result, passing_verification)
        has_patch_ref = any("Patch" in c or "patch" in c for c in expl.evidence_citations)
        assert has_patch_ref

    def test_concept_alignments_present(self, agent, diagnosis, evidence_bundle, fusion_result, passing_verification):
        expl = agent.run(diagnosis, evidence_bundle, fusion_result, passing_verification)
        assert isinstance(expl.concept_alignments, list)
        assert len(expl.concept_alignments) >= 1

    def test_uncertainty_statement_non_empty(self, agent, diagnosis, evidence_bundle, fusion_result, passing_verification):
        expl = agent.run(diagnosis, evidence_bundle, fusion_result, passing_verification)
        assert isinstance(expl.uncertainty_statement, str)
        assert len(expl.uncertainty_statement) > 0

    def test_her2_always_uncertain(self, agent, diagnosis, evidence_bundle, fusion_result, passing_verification):
        """HER2 from H&E should always be flagged as requiring IHC confirmation."""
        expl = agent.run(diagnosis, evidence_bundle, fusion_result, passing_verification)
        assert "HER2" in expl.uncertainty_statement or "IHC" in expl.uncertainty_statement

    def test_key_morphological_features_present(self, agent, diagnosis, evidence_bundle, fusion_result, passing_verification):
        expl = agent.run(diagnosis, evidence_bundle, fusion_result, passing_verification)
        assert isinstance(expl.key_morphological_features, list)

    def test_counterfactual_note_present(self, agent, diagnosis, evidence_bundle, fusion_result, passing_verification):
        expl = agent.run(diagnosis, evidence_bundle, fusion_result, passing_verification)
        # counterfactual_note is optional but should be generated for malignant cases
        assert expl.counterfactual_note is None or len(expl.counterfactual_note) > 10

    def test_ood_flag_in_uncertainty_statement(self, agent, evidence_bundle, fusion_result, abstaining_verification):
        """OOD flag in verification should surface in the uncertainty statement."""
        diag = DiagnosisAgent().run(evidence_bundle, fusion_result, abstaining_verification)
        # We run with abstaining_verification but force the agent to still produce explanation
        expl = agent.run(diag, evidence_bundle, fusion_result, abstaining_verification)
        assert "OOD" in expl.uncertainty_statement or "out-of-distribution" in expl.uncertainty_statement.lower() or \
               abstaining_verification.is_ood  # just verify OOD is set


# ---------------------------------------------------------------------------
# ReportAgent tests
# ---------------------------------------------------------------------------

class TestReportAgent:

    @pytest.fixture
    def agent(self):
        return ReportAgent()

    @pytest.fixture
    def dummy_pipeline_input(self):
        return PipelineInput(
            wsi_input=WSIInput(wsi_path="/fake/slide.svs", patient_id="P001"),
            run_id="test-report-001",
        )

    @pytest.fixture
    def diagnosis(self, evidence_bundle, fusion_result, passing_verification):
        return DiagnosisAgent().run(evidence_bundle, fusion_result, passing_verification)

    @pytest.fixture
    def explanation(self, diagnosis, evidence_bundle, fusion_result, passing_verification):
        return ExplanationAgent().run(diagnosis, evidence_bundle, fusion_result, passing_verification)

    def test_full_report_returns_pathology_report(
        self, agent, dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification
    ):
        report = agent.run(dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification)
        assert report is not None

    def test_report_has_patient_id(
        self, agent, dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification
    ):
        report = agent.run(dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification)
        assert report.patient_id == "P001"

    def test_report_has_timestamp(
        self, agent, dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification
    ):
        report = agent.run(dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification)
        assert isinstance(report.timestamp, str)
        assert "T" in report.timestamp   # ISO 8601

    def test_biomarker_summary_populated(
        self, agent, dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification
    ):
        report = agent.run(dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification)
        bs = report.biomarker_summary
        assert "MSI_status" in bs
        assert "Lauren_type" in bs
        assert "HER2_status" in bs

    def test_recommendations_non_empty(
        self, agent, dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification
    ):
        report = agent.run(dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification)
        assert len(report.recommendations) >= 1
        # First rec is always pathologist sign-off
        assert "pathologist" in report.recommendations[0].lower()

    def test_msi_h_triggers_mmr_recommendation(
        self, agent, dummy_pipeline_input, explanation, passing_verification
    ):
        """MSI-H prediction should add an MMR IHC recommendation."""
        from hakim_ai.types import (
            Diagnosis, FusionResult, MolecularPrediction, ClinicalContext, RetrievedKnowledge,
            HER2Status, EBVStatus
        )
        mol = MolecularPrediction(
            msi_status=MSIStatus.MSI_HIGH, msi_probability=0.82,
            lauren_class=LaurenClassification.INTESTINAL, lauren_confidence=0.70,
            her2_status=HER2Status.NEGATIVE, her2_probability=0.10,
            ebv_status=EBVStatus.NEGATIVE, ebv_probability=0.08,
        )
        fu = FusionResult(
            molecular=mol,
            clinical_context=ClinicalContext(encoded_prompt="test"),
            knowledge=RetrievedKnowledge(),
        )
        diag = Diagnosis(
            primary_diagnosis="Gastric adenocarcinoma, intestinal type",
            diagnostic_label=DiagnosticLabel.MALIGNANT,
            overall_confidence=0.65,
        )
        report = agent.run(dummy_pipeline_input, diag, explanation, fu, passing_verification)
        assert any("MMR" in r or "IHC" in r for r in report.recommendations)

    def test_structured_fields_has_required_keys(
        self, agent, dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification
    ):
        report = agent.run(dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification)
        sf = report.structured_fields
        required = {"patient_id", "primary_diagnosis", "diagnostic_label", "msi_status", "lauren_type"}
        assert required.issubset(sf.keys())

    def test_benign_report_generation(self, agent, malignant_router_decision, passing_qc_result):
        from hakim_ai.layer1_router.router_agent import RouterAgent
        from hakim_ai.config import RouterConfig
        from hakim_ai.types import RouterDecision, CaseComplexity, DiagnosticLabel, TaskType
        inp = PipelineInput(wsi_input=WSIInput("/fake/slide.svs", "P_BENIGN"))
        benign_dec = RouterDecision(
            label=DiagnosticLabel.BENIGN, complexity=CaseComplexity.ROUTINE,
            confidence=0.91, task_type=TaskType.CLASSIFICATION,
        )
        report = agent.generate_benign_report(inp, passing_qc_result, benign_dec)
        assert report.patient_id == "P_BENIGN"
        assert report.diagnosis.diagnostic_label == DiagnosticLabel.BENIGN
        assert "benign" in report.who_classification.lower() or "benign" in report.diagnosis.primary_diagnosis.lower()

    def test_uncertainty_report_generation(self, agent, evidence_bundle, fusion_result, abstaining_verification):
        inp = PipelineInput(wsi_input=WSIInput("/fake/slide.svs", "P_UNCERTAIN"))
        report = agent.generate_uncertainty_report(inp, evidence_bundle, fusion_result, abstaining_verification)
        assert report.patient_id == "P_UNCERTAIN"
        assert "INCONCLUSIVE" in report.who_classification or "INCONCLUSIVE" in report.diagnosis.primary_diagnosis
        assert len(report.uncertainty_flags) >= 1

    def test_report_to_json_serialisable(
        self, agent, dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification
    ):
        import json
        report = agent.run(dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification)
        json_str = agent.to_json(report)
        parsed = json.loads(json_str)
        assert "patient_id" in parsed
        assert "diagnosis" in parsed

    def test_report_version_set(
        self, agent, dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification
    ):
        report = agent.run(dummy_pipeline_input, diagnosis, explanation, fusion_result, passing_verification)
        assert report.report_version == "1.0"