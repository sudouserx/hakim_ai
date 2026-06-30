"""
Shared pytest fixtures for hakim_ai tests.

All fixtures use synthetic data — no real WSI files, model weights,
or GPU are required. The full test suite runs in <10 seconds.
"""
from __future__ import annotations

import pytest

from hakim_ai.config import PipelineConfig
from hakim_ai.types import (
    ClinicalInput,
    EvidenceBundle,
    FusionResult,
    LaurenClassification,
    LogicCheckResult,
    MolecularPrediction,
    MSIStatus,
    HER2Status,
    EBVStatus,
    NavigationResult,
    PatchCoordinate,
    PatchDescription,
    PipelineInput,
    QCResult,
    RetrievedKnowledge,
    RouterDecision,
    CaseComplexity,
    DiagnosticLabel,
    TaskType,
    TissueSegmentation,
    WHOValidationResult,
    ClinicalContext,
    WSIData,
    WSIInput,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def test_config() -> PipelineConfig:
    """Minimal test configuration — fast and deterministic."""
    return PipelineConfig.for_testing()


# ---------------------------------------------------------------------------
# Input fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def wsi_input() -> WSIInput:
    return WSIInput(
        wsi_path="/fake/slides/patient_001.svs",
        patient_id="P001",
        scanner_model="Aperio GT450",
        magnification=40.0,
    )


@pytest.fixture
def clinical_input() -> ClinicalInput:
    return ClinicalInput(
        patient_id="P001",
        age=62,
        sex="M",
        ehr_notes="Presenting with dysphagia and weight loss. Endoscopy reveals ulcerative lesion in gastric antrum.",
        endoscopy_findings="2.5cm ulcerative lesion, antrum. Biopsy taken.",
        biopsy_location="antrum",
        h_pylori_status=True,
        prior_treatments=[],
        molecular_reports="No prior molecular testing.",
    )


@pytest.fixture
def pipeline_input(wsi_input, clinical_input) -> PipelineInput:
    return PipelineInput(
        wsi_input=wsi_input,
        clinical_input=clinical_input,
        run_id="test-run-001",
    )


@pytest.fixture
def pipeline_input_no_clinical(wsi_input) -> PipelineInput:
    return PipelineInput(wsi_input=wsi_input, run_id="test-run-002")


@pytest.fixture
def pipeline_input_with_radiology(wsi_input, clinical_input) -> PipelineInput:
    return PipelineInput(
        wsi_input=wsi_input,
        clinical_input=clinical_input,
        radiology_path="/fake/ct/patient_001_ct.dcm",
        run_id="test-run-003",
    )


# ---------------------------------------------------------------------------
# WSIData fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def wsi_data(wsi_input) -> WSIData:
    return WSIData(
        patient_id=wsi_input.patient_id,
        wsi_path=wsi_input.wsi_path,
        thumbnail=[[[255, 255, 255] for _ in range(5)] for _ in range(5)],
        tile_paths=[],
        level_dimensions=[(1000, 1000)],
        level_count=1,
        mpp=0.25,
        metadata={},
        slide_handle=None,
    )


# ---------------------------------------------------------------------------
# QC fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def passing_qc_result() -> QCResult:
    return QCResult(
        passed=True,
        stain_quality_score=0.82,
        focus_quality_score=0.78,
        coverage_score=0.71,
        tissue_area_mm2=45.2,
        artifacts_detected=[],
        normalized_wsi_path="/fake/slides/patient_001_normalized.svs",
    )


@pytest.fixture
def failing_qc_result() -> QCResult:
    return QCResult(
        passed=False,
        stain_quality_score=0.30,
        focus_quality_score=0.25,
        coverage_score=0.15,
        artifacts_detected=["tissue_fold", "out_of_focus_region"],
        rejection_reason="Stain quality 0.30 < threshold 0.50; Focus quality 0.25 < threshold 0.50",
    )


# ---------------------------------------------------------------------------
# Router fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def malignant_router_decision() -> RouterDecision:
    return RouterDecision(
        label=DiagnosticLabel.MALIGNANT,
        complexity=CaseComplexity.INTERMEDIATE,
        confidence=0.72,
        task_type=TaskType.BIOMARKER_PREDICTION,
        escalate_to_human=False,
        routing_rationale="Synthetic malignant routing decision.",
    )


@pytest.fixture
def benign_router_decision() -> RouterDecision:
    return RouterDecision(
        label=DiagnosticLabel.BENIGN,
        complexity=CaseComplexity.ROUTINE,
        confidence=0.88,
        task_type=TaskType.CLASSIFICATION,
        escalate_to_human=False,
    )


# ---------------------------------------------------------------------------
# Evidence fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def patch_coords() -> list:
    return [
        PatchCoordinate(x=1024, y=2048, level=0, width=512, height=512, importance_score=0.91, label="tumour_core"),
        PatchCoordinate(x=3072, y=1024, level=1, width=512, height=512, importance_score=0.78, label="invasive_front"),
        PatchCoordinate(x=512,  y=4096, level=0, width=512, height=512, importance_score=0.65, label="stroma"),
    ]


@pytest.fixture
def navigation_result(patch_coords) -> NavigationResult:
    importance_map = [[float(i + j) / 31 for j in range(16)] for i in range(16)]
    return NavigationResult(
        selected_patches=patch_coords,
        importance_map=importance_map,
        diagnostic_regions=[
            {"region_id": "R1", "x": 1024, "y": 2048, "importance": 0.91, "label": "tumour_core", "level": 0},
        ],
        magnification_levels_used=[20, 40],
        top_patch_count=3,
    )


@pytest.fixture
def tissue_segmentation() -> TissueSegmentation:
    return TissueSegmentation(
        tumour_fraction=0.48,
        stroma_fraction=0.29,
        til_density=0.18,
        necrosis_fraction=0.04,
        normal_gland_fraction=0.07,
        tme_profile={
            "tumour_purity": 0.48,
            "stromal_fraction": 0.29,
            "immune_fraction": 0.18,
            "necrotic_fraction": 0.04,
            "til_density_score": 0.375,
        },
    )


@pytest.fixture
def patch_descriptions(patch_coords) -> list:
    return [
        PatchDescription(
            patch_coord=patch_coords[0],
            narrative=(
                "The patch demonstrates irregular glandular architecture with loss of "
                "normal mucosal organisation. Nuclei are enlarged with prominent nucleoli "
                "and increased nuclear-to-cytoplasmic ratio."
            ),
            morphological_features=["irregular glandular architecture", "enlarged nuclei with prominent nucleoli"],
            confidence=0.87,
            magnification=40.0,
        ),
        PatchDescription(
            patch_coord=patch_coords[1],
            narrative=(
                "A dense lymphocytic infiltrate surrounds tumour nests at the invasive margin. "
                "Tumour-infiltrating lymphocyte (TIL) density is high."
            ),
            morphological_features=["tumour-infiltrating lymphocytes (TILs)", "desmoplastic stromal reaction"],
            confidence=0.74,
            magnification=20.0,
        ),
    ]


@pytest.fixture
def evidence_bundle(navigation_result, tissue_segmentation, patch_descriptions) -> EvidenceBundle:
    return EvidenceBundle(
        navigation=navigation_result,
        segmentation=tissue_segmentation,
        descriptions=patch_descriptions,
    )


# ---------------------------------------------------------------------------
# Fusion fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def molecular_prediction() -> MolecularPrediction:
    return MolecularPrediction(
        msi_status=MSIStatus.MSI_HIGH,
        msi_probability=0.76,
        lauren_class=LaurenClassification.INTESTINAL,
        lauren_confidence=0.68,
        her2_status=HER2Status.NEGATIVE,
        her2_probability=0.12,
        ebv_status=EBVStatus.NEGATIVE,
        ebv_probability=0.15,
    )


@pytest.fixture
def clinical_context() -> ClinicalContext:
    return ClinicalContext(
        encoded_prompt="Patient: P001\nAge: 62, M\nBiopsy: antrum\nH. pylori: positive",
        relevant_history_items=["Age: 62 years", "Sex: M", "H. pylori infection history"],
        prior_molecular_results={},
        risk_factors=["H. pylori infection history"],
    )


@pytest.fixture
def retrieved_knowledge() -> RetrievedKnowledge:
    return RetrievedKnowledge(
        who_criteria=["Intestinal-type gastric adenocarcinoma: cohesive neoplastic cells forming gland-like structures."],
        guideline_passages=["MSI-H gastric carcinoma: Pembrolizumab eligible (KEYNOTE-059)."],
        similar_cases=[{"case_id": "TCGA-BR-4253", "diagnosis": "Gastric adenocarcinoma, intestinal type", "similarity_score": 0.87}],
        literature_evidence=["Kather et al. (Nature Medicine 2019): H&E-based MSI prediction (AUC 0.84)."],
    )


@pytest.fixture
def fusion_result(molecular_prediction, clinical_context, retrieved_knowledge) -> FusionResult:
    return FusionResult(
        molecular=molecular_prediction,
        clinical_context=clinical_context,
        knowledge=retrieved_knowledge,
    )


# ---------------------------------------------------------------------------
# Verification fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def passing_verification(molecular_prediction) -> VerificationResult:
    return VerificationResult(
        passed=True,
        logic_check=LogicCheckResult(consistent=True, issues=[], warnings=[]),
        who_validation=WHOValidationResult(
            compliant=True,
            matched_criteria=["tubular adenocarcinoma"],
            violations=[],
            suggested_classification="tubular adenocarcinoma",
        ),
        raw_confidence=0.74,
        calibrated_confidence=0.65,
        is_ood=False,
        abstain=False,
    )


@pytest.fixture
def abstaining_verification() -> VerificationResult:
    return VerificationResult(
        passed=False,
        logic_check=LogicCheckResult(
            consistent=False,
            issues=["MSI-H predicted but TIL density is very low."],
            warnings=[],
        ),
        who_validation=WHOValidationResult(compliant=True, matched_criteria=[], violations=[]),
        raw_confidence=0.28,
        calibrated_confidence=0.22,
        is_ood=True,
        abstain=True,
        abstention_reason="Calibrated confidence below threshold",
        uncertainty_source="feature_ambiguity",
    )