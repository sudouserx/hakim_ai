"""Unit tests for Layer 2 (Evidence Collection) and Layer 3 (Fusion)."""
from __future__ import annotations

import pytest

from hakim_ai.layer2_evidence import NavigationAgent, SegmentationAgent, DescriptionAgent
from hakim_ai.layer3_fusion import (
    MolecularPredictionAgent,
    ClinicalContextAgent,
    KnowledgeRetrievalAgent,
    RadiologyPathologyAgent,
)
from hakim_ai.foundation_models import build_patch_encoder, build_vlm
from hakim_ai.types import (
    LaurenClassification,
    MSIStatus,
    HER2Status,
    EBVStatus,
)


# ── Navigation Agent ─────────────────────────────────────────────────────────

class TestNavigationAgent:

    @pytest.fixture
    def agent(self, test_config):
        encoder = build_patch_encoder(test_config.foundation_models)
        return NavigationAgent(test_config.navigation, encoder)

    def test_returns_navigation_result(self, agent, wsi_data, passing_qc_result):
        result = agent.run(wsi_data, passing_qc_result)
        assert result is not None
        assert isinstance(result.selected_patches, list)

    def test_patches_within_configured_limit(self, agent, wsi_data, passing_qc_result, test_config):
        result = agent.run(wsi_data, passing_qc_result)
        assert len(result.selected_patches) <= test_config.navigation.top_k_patches

    def test_patches_have_importance_scores(self, agent, wsi_data, passing_qc_result):
        result = agent.run(wsi_data, passing_qc_result)
        for patch in result.selected_patches:
            assert 0.0 <= patch.importance_score <= 1.0

    def test_patches_sorted_by_importance(self, agent, wsi_data, passing_qc_result):
        result = agent.run(wsi_data, passing_qc_result)
        if len(result.selected_patches) >= 2:
            scores = [p.importance_score for p in result.selected_patches]
            assert scores == sorted(scores, reverse=True)

    def test_importance_map_shape(self, agent, wsi_data, passing_qc_result):
        result = agent.run(wsi_data, passing_qc_result)
        if result.importance_map:
            assert len(result.importance_map) == 16
            assert all(len(row) == 16 for row in result.importance_map)

    def test_importance_map_values_in_range(self, agent, wsi_data, passing_qc_result):
        result = agent.run(wsi_data, passing_qc_result)
        if result.importance_map:
            for row in result.importance_map:
                for val in row:
                    assert 0.0 <= val <= 1.0

    def test_magnification_levels_recorded(self, agent, wsi_data, passing_qc_result):
        result = agent.run(wsi_data, passing_qc_result)
        assert len(result.magnification_levels_used) > 0

    def test_diagnostic_regions_present(self, agent, wsi_data, passing_qc_result):
        result = agent.run(wsi_data, passing_qc_result)
        assert isinstance(result.diagnostic_regions, list)


# ── Segmentation Agent ───────────────────────────────────────────────────────

class TestSegmentationAgent:

    @pytest.fixture
    def agent(self, test_config):
        return SegmentationAgent(test_config.segmentation)

    def test_returns_segmentation_result(self, agent, wsi_data, navigation_result):
        result = agent.run(wsi_data, navigation_result)
        assert result is not None

    def test_fractions_sum_approx_one(self, agent, wsi_data, navigation_result):
        result = agent.run(wsi_data, navigation_result)
        total = (
            result.tumour_fraction
            + result.stroma_fraction
            + result.til_density
            + result.necrosis_fraction
            + result.normal_gland_fraction
        )
        # Not required to exactly sum to 1 (background not included), but should be <1
        assert total <= 1.0 + 0.01

    def test_fractions_non_negative(self, agent, wsi_data, navigation_result):
        result = agent.run(wsi_data, navigation_result)
        assert result.tumour_fraction >= 0.0
        assert result.stroma_fraction >= 0.0
        assert result.til_density >= 0.0
        assert result.necrosis_fraction >= 0.0
        assert result.normal_gland_fraction >= 0.0

    def test_tme_profile_populated(self, agent, wsi_data, navigation_result):
        result = agent.run(wsi_data, navigation_result)
        assert isinstance(result.tme_profile, dict)
        assert "tumour_purity" in result.tme_profile

    def test_region_labels_match_navigation(self, agent, wsi_data, navigation_result):
        result = agent.run(wsi_data, navigation_result)
        assert len(result.region_labels) == len(navigation_result.diagnostic_regions)


# ── Description Agent ─────────────────────────────────────────────────────────

class TestDescriptionAgent:

    @pytest.fixture
    def agent(self, test_config):
        vlm = build_vlm(test_config.foundation_models)
        return DescriptionAgent(test_config.description, vlm)

    def test_returns_list_of_descriptions(self, agent, wsi_data, navigation_result, test_config):
        descriptions = agent.run(wsi_data, navigation_result)
        assert isinstance(descriptions, list)
        assert len(descriptions) <= test_config.description.max_patches_to_describe

    def test_each_description_has_narrative(self, agent, wsi_data, navigation_result):
        descriptions = agent.run(wsi_data, navigation_result)
        for desc in descriptions:
            assert isinstance(desc.narrative, str)
            assert len(desc.narrative) > 10

    def test_each_description_has_features(self, agent, wsi_data, navigation_result):
        descriptions = agent.run(wsi_data, navigation_result)
        for desc in descriptions:
            assert isinstance(desc.morphological_features, list)
            assert len(desc.morphological_features) >= 1

    def test_confidence_in_range(self, agent, wsi_data, navigation_result):
        descriptions = agent.run(wsi_data, navigation_result)
        for desc in descriptions:
            assert 0.0 <= desc.confidence <= 1.0


# ── Molecular Prediction Agent ────────────────────────────────────────────────

class TestMolecularPredictionAgent:

    @pytest.fixture
    def agent(self, test_config):
        return MolecularPredictionAgent(test_config.molecular)

    def test_returns_molecular_prediction(self, agent, evidence_bundle):
        pred = agent.run(evidence_bundle)
        assert pred is not None

    def test_msi_probability_in_range(self, agent, evidence_bundle):
        pred = agent.run(evidence_bundle)
        assert 0.0 <= pred.msi_probability <= 1.0

    def test_her2_probability_in_range(self, agent, evidence_bundle):
        pred = agent.run(evidence_bundle)
        assert 0.0 <= pred.her2_probability <= 1.0

    def test_ebv_probability_in_range(self, agent, evidence_bundle):
        pred = agent.run(evidence_bundle)
        assert 0.0 <= pred.ebv_probability <= 1.0

    def test_msi_status_consistent_with_probability(self, agent, evidence_bundle):
        pred = agent.run(evidence_bundle)
        if pred.msi_status == MSIStatus.MSI_HIGH:
            assert pred.msi_probability >= 0.5
        else:
            assert pred.msi_probability < 0.5

    def test_lauren_confidence_in_range(self, agent, evidence_bundle):
        pred = agent.run(evidence_bundle)
        assert 0.0 <= pred.lauren_confidence <= 1.0

    def test_lauren_class_is_valid_enum(self, agent, evidence_bundle):
        pred = agent.run(evidence_bundle)
        assert pred.lauren_class in list(LaurenClassification)

    def test_high_til_predicts_msi_high_more_often(self, test_config):
        """TIL density > 0.3 should increase MSI-H probability."""
        from hakim_ai.types import TissueSegmentation, NavigationResult, EvidenceBundle
        agent = MolecularPredictionAgent(test_config.molecular, seed=123)

        low_til_seg = TissueSegmentation(til_density=0.02, tumour_fraction=0.4, stroma_fraction=0.3)
        high_til_seg = TissueSegmentation(til_density=0.45, tumour_fraction=0.4, stroma_fraction=0.3)

        nav = NavigationResult(selected_patches=[], top_patch_count=0)
        low_bundle = EvidenceBundle(navigation=nav, segmentation=low_til_seg, descriptions=[])
        high_bundle = EvidenceBundle(navigation=nav, segmentation=high_til_seg, descriptions=[])

        low_pred = agent.run(low_bundle)
        high_pred = agent.run(high_bundle)
        assert high_pred.msi_probability > low_pred.msi_probability


# ── Clinical Context Agent ────────────────────────────────────────────────────

class TestClinicalContextAgent:

    @pytest.fixture
    def agent(self):
        return ClinicalContextAgent()

    def test_returns_context_with_clinical_data(self, agent, clinical_input):
        ctx = agent.run(clinical_input)
        assert ctx is not None
        assert isinstance(ctx.encoded_prompt, str)
        assert len(ctx.encoded_prompt) > 0

    def test_empty_context_on_none(self, agent):
        ctx = agent.run(None)
        assert ctx is not None
        assert "No clinical data" in ctx.encoded_prompt

    def test_age_and_sex_in_prompt(self, agent, clinical_input):
        ctx = agent.run(clinical_input)
        assert "62" in ctx.encoded_prompt or "Age" in ctx.encoded_prompt

    def test_risk_factors_extracted(self, agent, clinical_input):
        ctx = agent.run(clinical_input)
        assert len(ctx.risk_factors) >= 1   # H. pylori should appear

    def test_prior_molecular_parsed(self, agent):
        from hakim_ai.types import ClinicalInput
        ci = ClinicalInput(
            patient_id="P999",
            molecular_reports="MSI-H status confirmed by MMR immunohistochemistry.",
        )
        ctx = agent.run(ci)
        assert "MSI" in ctx.prior_molecular_results


# ── Knowledge Retrieval Agent ─────────────────────────────────────────────────

class TestKnowledgeRetrievalAgent:

    @pytest.fixture
    def agent(self, test_config):
        return KnowledgeRetrievalAgent(test_config.rag)

    def test_returns_retrieved_knowledge(self, agent, molecular_prediction, evidence_bundle):
        knowledge = agent.run(molecular_prediction, evidence_bundle)
        assert knowledge is not None

    def test_who_criteria_populated(self, agent, molecular_prediction, evidence_bundle):
        knowledge = agent.run(molecular_prediction, evidence_bundle)
        assert isinstance(knowledge.who_criteria, list)
        assert len(knowledge.who_criteria) >= 1

    def test_similar_cases_populated(self, agent, molecular_prediction, evidence_bundle):
        knowledge = agent.run(molecular_prediction, evidence_bundle)
        assert isinstance(knowledge.similar_cases, list)

    def test_literature_for_msi_high(self, agent, evidence_bundle):
        from hakim_ai.types import (
            MolecularPrediction, LaurenClassification, MSIStatus, HER2Status, EBVStatus
        )
        msi_h_pred = MolecularPrediction(
            msi_status=MSIStatus.MSI_HIGH, msi_probability=0.82,
            lauren_class=LaurenClassification.INTESTINAL, lauren_confidence=0.7,
            her2_status=HER2Status.NEGATIVE, her2_probability=0.1,
            ebv_status=EBVStatus.NEGATIVE, ebv_probability=0.1,
        )
        knowledge = agent.run(msi_h_pred, evidence_bundle)
        assert any("MSI" in lit or "Kather" in lit for lit in knowledge.literature_evidence)


# ── Radiology Agent ───────────────────────────────────────────────────────────

class TestRadiologyAgent:

    def test_returns_none_for_empty_path(self, evidence_bundle):
        agent = RadiologyPathologyAgent()
        result = agent.run("", evidence_bundle)
        assert result is None

    def test_returns_string_for_valid_path(self, evidence_bundle):
        agent = RadiologyPathologyAgent()
        result = agent.run("/fake/ct.dcm", evidence_bundle)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_summary_mentions_radiology(self, evidence_bundle):
        agent = RadiologyPathologyAgent()
        result = agent.run("/fake/ct.dcm", evidence_bundle)
        assert "radiology" in result.lower() or "ct" in result.lower() or "mri" in result.lower()