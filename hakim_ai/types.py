"""
Core typed data structures for the histopathology AI pipeline.

All inter-layer communication uses these types, enabling clean interfaces,
testability, and future model swapping without touching downstream code.

Assumption: numpy arrays (masks, importance maps) are represented as
List[List[float]] here to avoid a hard numpy dependency in the type
module. Real adapters swap these for np.ndarray.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DiagnosticLabel(str, Enum):
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    MALIGNANT = "malignant"
    UNKNOWN = "unknown"


class LaurenClassification(str, Enum):
    INTESTINAL = "intestinal"
    DIFFUSE = "diffuse"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class CaseComplexity(str, Enum):
    ROUTINE = "routine"
    INTERMEDIATE = "intermediate"
    COMPLEX = "complex"


class MSIStatus(str, Enum):
    MSI_HIGH = "MSI-H"
    MSS = "MSS"
    UNKNOWN = "unknown"


class HER2Status(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    EQUIVOCAL = "equivocal"
    UNKNOWN = "unknown"


class EBVStatus(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    BIOMARKER_PREDICTION = "biomarker_prediction"
    REPORT_GENERATION = "report_generation"
    SURVIVAL_PREDICTION = "survival_prediction"


# ---------------------------------------------------------------------------
# Pipeline inputs
# ---------------------------------------------------------------------------

@dataclass
class WSIInput:
    """Whole-slide image input with scan metadata."""
    wsi_path: str
    patient_id: str
    scanner_model: Optional[str] = None
    magnification: float = 40.0
    scan_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClinicalInput:
    """Structured patient clinical data and EHR notes."""
    patient_id: str
    age: Optional[int] = None
    sex: Optional[str] = None          # "M" | "F" | None
    ehr_notes: Optional[str] = None
    molecular_reports: Optional[str] = None
    endoscopy_findings: Optional[str] = None
    prior_treatments: List[str] = field(default_factory=list)
    h_pylori_status: Optional[bool] = None
    family_history: Optional[str] = None
    biopsy_location: Optional[str] = None   # antrum | body | cardia


@dataclass
class PipelineInput:
    """Top-level container delivered to the pipeline entry point."""
    wsi_input: WSIInput
    clinical_input: Optional[ClinicalInput] = None
    radiology_path: Optional[str] = None   # CT/MRI path if available
    run_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal WSI representation (passed between layer 0 agents)
# ---------------------------------------------------------------------------

@dataclass
class WSIData:
    """Loaded WSI representation (tiles + metadata)."""
    patient_id: str
    wsi_path: str
    # thumbnail as H×W×3 list; real impl uses np.ndarray
    thumbnail: Optional[List[List[List[int]]]] = None
    tile_paths: List[str] = field(default_factory=list)
    level_dimensions: List[tuple] = field(default_factory=list)  # [(W, H), ...]
    level_count: int = 1
    mpp: float = 0.25   # microns-per-pixel at highest magnification
    metadata: Dict[str, Any] = field(default_factory=dict)
    slide_handle: Any = None  # Persistent OpenSlide handle for patch extraction


# ---------------------------------------------------------------------------
# Layer 0: QC outputs
# ---------------------------------------------------------------------------

@dataclass
class QCResult:
    """Output from the quality control agent."""
    passed: bool
    stain_quality_score: float      # 0–1; 1 = perfect
    focus_quality_score: float      # 0–1
    coverage_score: float           # 0–1; fraction of slide covered by tissue
    tissue_area_mm2: float = 0.0
    artifacts_detected: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None
    normalized_wsi_path: Optional[str] = None
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layer 1: Router/triage outputs
# ---------------------------------------------------------------------------

@dataclass
class RouterDecision:
    """Routing and triage decision returned to the orchestrator."""
    label: DiagnosticLabel
    complexity: CaseComplexity
    confidence: float               # 0–1 router's confidence in its label
    task_type: TaskType
    escalate_to_human: bool = False
    routing_rationale: str = ""
    skip_radiology_fusion: bool = False


# ---------------------------------------------------------------------------
# Layer 2: Evidence collection outputs
# ---------------------------------------------------------------------------

@dataclass
class PatchCoordinate:
    """Spatial address of a single WSI patch."""
    x: int
    y: int
    level: int          # 0 = highest magnification level
    width: int = 512
    height: int = 512
    importance_score: float = 0.0
    label: Optional[str] = None     # e.g. "tumour_core", "invasive_front"
    feature_vector: Optional[List[float]] = None


@dataclass
class NavigationResult:
    """Multi-scale patch selection and importance mapping."""
    selected_patches: List[PatchCoordinate]
    # importance_map: 2-D grid of floats at thumbnail resolution
    importance_map: Optional[List[List[float]]] = None
    diagnostic_regions: List[Dict[str, Any]] = field(default_factory=list)
    magnification_levels_used: List[int] = field(default_factory=list)
    top_patch_count: int = 0


@dataclass
class TissueSegmentation:
    """Per-compartment tissue fractions and TME profile."""
    tumour_fraction: float = 0.0
    stroma_fraction: float = 0.0
    til_density: float = 0.0        # tumour-infiltrating lymphocyte density 0–1
    necrosis_fraction: float = 0.0
    normal_gland_fraction: float = 0.0
    tme_profile: Dict[str, float] = field(default_factory=dict)
    region_labels: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PatchDescription:
    """Natural-language description of a single diagnostic patch."""
    patch_coord: PatchCoordinate
    narrative: str
    morphological_features: List[str] = field(default_factory=list)
    confidence: float = 0.0
    magnification: float = 20.0


@dataclass
class EvidenceBundle:
    """Aggregated evidence from all Layer 2 agents."""
    navigation: NavigationResult
    segmentation: TissueSegmentation
    descriptions: List[PatchDescription]
    patch_feature_vectors: Dict[str, List[float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Layer 3: Multimodal fusion outputs
# ---------------------------------------------------------------------------

@dataclass
class MolecularPrediction:
    """H&E-predicted biomarker status for gastric/STAD."""
    msi_status: MSIStatus
    msi_probability: float          # P(MSI-H)
    lauren_class: LaurenClassification
    lauren_confidence: float
    her2_status: HER2Status
    her2_probability: float         # P(HER2+)
    ebv_status: EBVStatus
    ebv_probability: float          # P(EBV+)
    raw_logits: Dict[str, float] = field(default_factory=dict)


@dataclass
class ClinicalContext:
    """Encoded clinical context ready for LLM fusion."""
    encoded_prompt: str
    relevant_history_items: List[str] = field(default_factory=list)
    prior_molecular_results: Dict[str, str] = field(default_factory=dict)
    risk_factors: List[str] = field(default_factory=list)
    cross_modal_relevance: float = 0.5


@dataclass
class RetrievedKnowledge:
    """Knowledge base retrieval results (RAG output)."""
    who_criteria: List[str] = field(default_factory=list)
    guideline_passages: List[str] = field(default_factory=list)
    similar_cases: List[Dict[str, Any]] = field(default_factory=list)
    literature_evidence: List[str] = field(default_factory=list)
    retrieval_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class FusionResult:
    """Aggregated output from all Layer 3 agents."""
    molecular: MolecularPrediction
    clinical_context: ClinicalContext
    knowledge: RetrievedKnowledge
    radiology_findings: Optional[str] = None


# ---------------------------------------------------------------------------
# Layer 4: Verification outputs
# ---------------------------------------------------------------------------

@dataclass
class LogicCheckResult:
    """Output of the internal consistency checker."""
    consistent: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class WHOValidationResult:
    """Output of the WHO taxonomy/criteria validator."""
    compliant: bool
    matched_criteria: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    suggested_classification: Optional[str] = None


@dataclass
class VerificationResult:
    """Unified verification and confidence calibration output."""
    passed: bool
    logic_check: LogicCheckResult
    who_validation: WHOValidationResult
    raw_confidence: float = 0.0
    calibrated_confidence: float = 0.0
    is_ood: bool = False            # out-of-distribution flag
    abstain: bool = False
    abstention_reason: Optional[str] = None
    # where uncertainty comes from: "slide_quality" | "feature_ambiguity" | "rare_subtype"
    uncertainty_source: Optional[str] = None
    consistency_issues: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Layer 5: Synthesis outputs
# ---------------------------------------------------------------------------

@dataclass
class Diagnosis:
    """Holistic final diagnosis."""
    primary_diagnosis: str
    diagnostic_label: DiagnosticLabel
    differential_diagnoses: List[str] = field(default_factory=list)
    grade: Optional[str] = None         # e.g. "poorly differentiated"
    tnm_contribution: Optional[str] = None
    overall_confidence: float = 0.0
    supporting_findings: List[str] = field(default_factory=list)


@dataclass
class Explanation:
    """Clinician-facing natural language explanation."""
    narrative: str
    evidence_citations: List[str] = field(default_factory=list)
    concept_alignments: List[str] = field(default_factory=list)  # WHO criteria links
    uncertainty_statement: str = ""
    key_morphological_features: List[str] = field(default_factory=list)
    counterfactual_note: Optional[str] = None


@dataclass
class PathologyReport:
    """Structured pathology report — the pipeline's primary deliverable."""
    patient_id: str
    timestamp: str
    diagnosis: Diagnosis
    molecular_predictions: MolecularPrediction
    explanation: Explanation
    lauren_classification: LaurenClassification
    who_classification: str
    uncertainty_flags: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    structured_fields: Dict[str, Any] = field(default_factory=dict)
    biomarker_summary: Dict[str, str] = field(default_factory=dict)
    report_version: str = "1.0"


# ---------------------------------------------------------------------------
# Top-level pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Complete output of the end-to-end pipeline run."""
    patient_id: str
    run_id: Optional[str]
    qc_result: QCResult
    router_decision: RouterDecision
    evidence: Optional[EvidenceBundle] = None
    fusion: Optional[FusionResult] = None
    verification: Optional[VerificationResult] = None
    report: Optional[PathologyReport] = None
    escalated_to_human: bool = False
    error: Optional[str] = None
    pipeline_duration_seconds: float = 0.0

    def is_successful(self) -> bool:
        return self.error is None and self.report is not None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)