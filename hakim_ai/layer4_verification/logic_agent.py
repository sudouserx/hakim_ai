"""
Layer 4 — Logic Agent.

Checks for internal contradictions between the evidence collected in
Layers 2 and 3. Contradictions that would make a diagnosis
clinically implausible are flagged before the Diagnosis Agent runs.

Based on the WSI-Agents (MICCAI 2025) dual-path verification pattern:
  (1) Internal consistency check   ← this module
  (2) External knowledge validation ← WHO Validator

Design principle (from architecture doc):
  "Avoid sycophantic agents. In multi-agent debates, agents can converge
   to consensus too quickly. Design verification agents to actively
   challenge the diagnosis agent."
"""
from __future__ import annotations

from typing import List

from hakim_ai.types import (
    EvidenceBundle,
    FusionResult,
    LaurenClassification,
    LogicCheckResult,
    MSIStatus,
)
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer4.logic")


# ---------------------------------------------------------------------------
# Consistency rules
# Tuples of (rule_name, check_fn, issue_message)
# check_fn returns True if the rule is VIOLATED (i.e., inconsistent)
# ---------------------------------------------------------------------------

def _rule_poorly_diff_not_intestinal(ev: EvidenceBundle, fu: FusionResult) -> bool:
    """Poorly differentiated histology and intestinal Lauren type is unusual."""
    has_poor_diff = any(
        "poorly differentiated" in d.narrative.lower()
        or "poor differentiation" in d.narrative.lower()
        for d in ev.descriptions
    )
    is_intestinal = fu.molecular.lauren_class == LaurenClassification.INTESTINAL
    return has_poor_diff and is_intestinal and fu.molecular.lauren_confidence > 0.80


def _rule_msi_low_til(ev: EvidenceBundle, fu: FusionResult) -> bool:
    """MSI-H with very low TIL density is biologically inconsistent."""
    return (
        fu.molecular.msi_status == MSIStatus.MSI_HIGH
        and ev.segmentation.til_density < 0.05
    )


def _rule_ebv_no_lymphoid(ev: EvidenceBundle, fu: FusionResult) -> bool:
    """EBV-positive gastric cancer without any lymphoid infiltrate is unusual."""
    if fu.molecular.ebv_probability < 0.50:
        return False
    has_lymphoid = any(
        "lympho" in d.narrative.lower() or "lymphocyte" in d.narrative.lower()
        for d in ev.descriptions
    )
    return not has_lymphoid


def _rule_her2_diffuse(ev: EvidenceBundle, fu: FusionResult) -> bool:
    """HER2 positivity is rare in diffuse-type gastric cancer (<5%)."""
    return (
        fu.molecular.her2_status.value == "positive"
        and fu.molecular.lauren_class == LaurenClassification.DIFFUSE
        and fu.molecular.lauren_confidence > 0.75
    )


def _rule_no_tumour_high_confidence(ev: EvidenceBundle, fu: FusionResult) -> bool:
    """Very low tumour fraction with high-confidence malignant molecular predictions."""
    return (
        ev.segmentation.tumour_fraction < 0.05
        and fu.molecular.msi_probability > 0.85
    )


def _rule_radiology_discordance(ev: EvidenceBundle, fu: FusionResult) -> bool:
    """Advanced T-stage on radiology but minimal tumour on histology (biopsy sampling error)."""
    if fu.radiology_findings is None:
        return False
    return (
        "T3" in fu.radiology_findings or "T4" in fu.radiology_findings
    ) and ev.segmentation.tumour_fraction < 0.1


def _rule_clinical_molecular_discordance(ev: EvidenceBundle, fu: FusionResult) -> bool:
    """Prior molecular results contradict current prediction."""
    prior = fu.clinical_context.prior_molecular_results
    if "MSI" in prior:
        if "MSS" in prior["MSI"] and fu.molecular.msi_status == MSIStatus.MSI_HIGH:
            return True
        if "MSI-H" in prior["MSI"] and fu.molecular.msi_status == MSIStatus.MSS:
            return True
    return False


_RULES = [
    (
        "poorly_diff_intestinal_inconsistency",
        _rule_poorly_diff_not_intestinal,
        "Poorly differentiated morphology with high-confidence intestinal Lauren type — "
        "consider mixed or diffuse Lauren subtype.",
    ),
    (
        "msi_h_without_til",
        _rule_msi_low_til,
        "MSI-H predicted but TIL density is very low (<5%). "
        "MSI-H is typically associated with prominent TIL infiltrate. "
        "Recommend IHC MMR testing (MLH1/MSH2/MSH6/PMS2).",
    ),
    (
        "ebv_without_lymphoid_stroma",
        _rule_ebv_no_lymphoid,
        "EBV probability >50% but no lymphoid infiltrate described in patch narratives. "
        "EBER-ISH recommended to confirm EBV status.",
    ),
    (
        "her2_positive_diffuse_type",
        _rule_her2_diffuse,
        "HER2 positivity predicted in diffuse-type Lauren tumour — "
        "this combination is rare. HER2 IHC recommended for confirmation.",
    ),
    (
        "low_tumour_fraction_high_confidence",
        _rule_no_tumour_high_confidence,
        "Very low tumour fraction on segmentation but high molecular prediction "
        "confidence — possible sampling artefact or intratumoural heterogeneity.",
    ),
    (
        "radiology_histology_discordance",
        _rule_radiology_discordance,
        "Radiology suggests advanced T-stage (T3/T4) but biopsy shows <10% tumour. "
        "Possible sampling error; superficial biopsy may not represent deepest invasion.",
    ),
    (
        "clinical_molecular_discordance",
        _rule_clinical_molecular_discordance,
        "Current molecular prediction contradicts prior molecular testing results from clinical history. "
        "Recommend re-testing or reviewing prior reports.",
    ),
]


class LogicAgent:
    """
    Internal consistency checker.

    Inputs:  EvidenceBundle, FusionResult
    Outputs: LogicCheckResult
    """

    def run(
        self, evidence: EvidenceBundle, fusion: FusionResult
    ) -> LogicCheckResult:
        logger.info("Logic consistency check started")

        issues: List[str] = []
        warnings: List[str] = []

        for rule_name, check_fn, message in _RULES:
            try:
                if check_fn(evidence, fusion):
                    logger.warning("Logic rule violated: %s", rule_name)
                    issues.append(message)
            except Exception as exc:
                warnings.append(f"Rule '{rule_name}' check failed with: {exc}")

        # Soft warnings (non-blocking)
        if not evidence.descriptions:
            warnings.append("No patch descriptions available — explanation quality may be reduced.")
        if evidence.navigation.top_patch_count < 5:
            warnings.append(
                f"Only {evidence.navigation.top_patch_count} patches selected — "
                "consider increasing top_k_patches for complex cases."
            )

        consistent = len(issues) == 0
        if consistent:
            logger.info("Logic check PASSED — no consistency issues found")
        else:
            logger.warning("Logic check: %d issue(s) found", len(issues))

        return LogicCheckResult(
            consistent=consistent,
            issues=issues,
            warnings=warnings,
        )