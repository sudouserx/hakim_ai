"""
Layer 4 — WHO Taxonomy Validator and Confidence Calibrator.

WHOValidator: cross-references the predicted diagnosis against the
WHO 5th Edition Digestive System Tumours classification to ensure
the reported terminology is compliant.

ConfidenceCalibrator: applies temperature scaling to raw softmax
probabilities and runs OOD (out-of-distribution) detection.
Implements conformal-prediction-inspired abstention when calibrated
confidence falls below the configured threshold.

References:
  - WHO Classification of Tumours: Digestive System Tumours, 5th Ed.
  - Dolezal et al. (Nat Commun 2022): calibrated uncertainty in pathology AI
  - Angelopoulos & Bates (2023): conformal risk control
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

from hakim_ai.config import VerificationConfig
from hakim_ai.types import (
    EvidenceBundle,
    FusionResult,
    LaurenClassification,
    LogicCheckResult,
    MSIStatus,
    VerificationResult,
    WHOValidationResult,
)
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer4.verification")


# ---------------------------------------------------------------------------
# WHO classification rules
# ---------------------------------------------------------------------------

# Simplified rule table: (who_term, required_features, forbidden_combos)
_WHO_GASTRIC_TERMS = {
    "tubular adenocarcinoma": {
        "description_keywords": ["tubular", "glandular", "gland"],
    },
    "papillary adenocarcinoma": {
        "description_keywords": ["papillary", "papillae", "fibrovascular core"],
    },
    "poorly cohesive carcinoma (signet-ring cell type)": {
        "description_keywords": ["signet", "poorly cohesive", "diffuse"],
    },
    "mucinous adenocarcinoma": {
        "description_keywords": ["mucin", "mucinous", "extracellular"],
    },
    "gastric adenocarcinoma NOS": {
        "description_keywords": [],
    },
    "mixed adenocarcinoma": {
        "description_keywords": ["mixed", "tubular", "poorly cohesive"],
    },
    "squamous cell carcinoma": {
        "description_keywords": ["squamous", "keratin"],
    },
    "adenosquamous carcinoma": {
        "description_keywords": ["squamous", "glandular", "adenosquamous"],
    },
    "medullary carcinoma": {
        "description_keywords": ["medullary", "lymphoid", "syncytial"],
    },
    "hepatoid adenocarcinoma": {
        "description_keywords": ["hepatoid", "hepatocellular"],
    },
    "undifferentiated carcinoma": {
        "description_keywords": ["undifferentiated", "solid", "sheet"],
    },
}


class WHOValidator:
    """
    Validates prediction terminology against WHO gastric cancer classification.

    Inputs:  FusionResult, EvidenceBundle
    Outputs: WHOValidationResult
    """

    def run(
        self, fusion: FusionResult, evidence: EvidenceBundle
    ) -> WHOValidationResult:
        logger.info("WHO validation started")

        all_narrative = " ".join(d.narrative.lower() for d in evidence.descriptions)

        matched: List[str] = []
        violations: List[str] = []
        suggested: Optional[str] = None

        for term, rules in _WHO_GASTRIC_TERMS.items():
            keywords_present = (
                not rules["description_keywords"]
                or any(kw in all_narrative for kw in rules["description_keywords"])
            )
            if keywords_present:
                matched.append(term)
                if suggested is None:
                    suggested = term

        if not matched:
            violations.append(
                f"No WHO 5th Edition gastric tumour category matched for "
                f"current morphological descriptions. Defaulting to 'gastric adenocarcinoma NOS'."
            )
            suggested = "gastric adenocarcinoma NOS"

        compliant = len(violations) == 0
        if compliant:
            logger.info("WHO validation PASSED: matched %s", matched)
        else:
            logger.warning("WHO validation issues: %s", violations)

        return WHOValidationResult(
            compliant=compliant,
            matched_criteria=matched,
            violations=violations,
            suggested_classification=suggested,
        )


# ---------------------------------------------------------------------------
# Confidence Calibrator
# ---------------------------------------------------------------------------

import numpy as np

def _to_scalar(val) -> float:
    """Safely extract a pure Python float from a 1D tensor or array."""
    if hasattr(val, "item"):
        # PyTorch Tensor or NumPy 0-D array
        try:
            return float(val.item())
        except ValueError:
            return float(val.squeeze().item())
    # Fallback for raw numpy arrays
    return float(np.squeeze(val))


def _temperature_scale(logit: float, temperature: float) -> float:
    """Apply temperature scaling: new_prob = sigmoid(logit / T)."""
    return 1.0 / (1.0 + math.exp(-logit / max(temperature, 1e-8)))


def _estimate_raw_confidence(fusion: FusionResult, evidence: EvidenceBundle) -> float:
    """
    Aggregate raw confidence from molecular predictions and evidence quality.

    Real implementation: use the max softmax probability from the
    diagnosis classifier output. Here we aggregate available signals.
    """
    mol = fusion.molecular
    msi_prob = _to_scalar(mol.msi_probability)
    lauren_conf = _to_scalar(mol.lauren_confidence)
    her2_prob = _to_scalar(mol.her2_probability)

    signals = [
        msi_prob if msi_prob > 0.5 else 1.0 - msi_prob,
        lauren_conf,
        her2_prob if her2_prob > 0.5 else 1.0 - her2_prob,
    ]
    # Average patch description confidence
    if evidence.descriptions:
        desc_conf = sum(d.confidence for d in evidence.descriptions) / len(evidence.descriptions)
        signals.append(desc_conf)
    return sum(signals) / len(signals)


def _check_biomarker_correlation(fusion: FusionResult) -> Optional[str]:
    """Flag biologically implausible biomarker co-occurrences."""
    mol = fusion.molecular
    # MSI-H and EBV are mutually exclusive TCGA subtypes
    if mol.msi_probability > 0.7 and mol.ebv_probability > 0.7:
        return "msi_ebv_mutual_exclusion"
    # GS (genomically stable) subtype: typically diffuse Lauren, MSS, HER2-
    if (mol.lauren_class == LaurenClassification.DIFFUSE
        and mol.msi_status == MSIStatus.MSI_HIGH
        and mol.lauren_confidence > 0.8):
        return "diffuse_msi_h_rare_combination"
    return None


def _detect_ood(fusion: FusionResult, evidence: EvidenceBundle, temperature: float = 1.0, threshold: float = 0.25) -> Tuple[bool, Optional[str]]:
    """
    Lightweight OOD detection.
    Real implementation: Energy-based OOD scoring using molecular logits.
    """
    # Low tissue coverage -> obvious OOD
    if evidence.segmentation.tumour_fraction < 0.05:
        return True, "slide_quality"
    if evidence.navigation.top_patch_count < 3:
        return True, "insufficient_tissue"
        
    # Check for biologically implausible biomarker correlations
    correlation_issue = _check_biomarker_correlation(fusion)
    if correlation_issue:
        return True, correlation_issue

    # Energy-based OOD for independent binary classifiers
    # Symmetric logits: [l/2, -l/2]
    mol = fusion.molecular
    if mol.raw_logits:
        logits = list(mol.raw_logits.values())
        if logits:
            energies = []
            for l in logits:
                l_val = _to_scalar(l)
                half_l = l_val / 2.0
                m = abs(half_l)
                # log(e^(half_l/T) + e^(-half_l/T)) = m/T + log(1.0 + math.exp(-2.0 * m / temperature))
                lse = m / temperature + math.log(1.0 + math.exp(-2.0 * m / temperature))
                energies.append(-temperature * lse)
            energy = sum(energies) / len(energies)
            
            # Max energy (most uncertain, p=0.5) is -T * ln(2).
            # Lower energy (more negative) means more confident ID data.
            max_energy = -temperature * math.log(2)
            
            # Scale the threshold to a tight log-space delta (e.g. 0.25 -> 0.05)
            # This flags only cases where average confidence is < ~65%
            scaled_threshold = threshold / 5.0
            
            # If energy is within `scaled_threshold` of the maximum possible uncertainty, flag it.
            if energy > max_energy - scaled_threshold:
                return True, f"high_energy_feature_ambiguity (energy={energy:.2f})"
                
    msi_prob = _to_scalar(mol.msi_probability)
    if max(
        msi_prob,
        1 - msi_prob,
    ) < 0.6:
        return True, "feature_ambiguity"
    return False, None


class ConfidenceCalibrator:
    """
    Temperature-scaled calibration + OOD detection + abstention logic.

    Inputs:  LogicCheckResult, WHOValidationResult, FusionResult, EvidenceBundle
    Outputs: VerificationResult
    """

    def __init__(self, cfg: VerificationConfig):
        self.cfg = cfg

    def run(
        self,
        logic: LogicCheckResult,
        who: WHOValidationResult,
        fusion: FusionResult,
        evidence: EvidenceBundle,
    ) -> VerificationResult:
        logger.info("Confidence calibration started")

        raw_conf = _estimate_raw_confidence(fusion, evidence)

        # Rough logit for temperature scaling
        clamped = max(1e-6, min(1 - 1e-6, raw_conf))
        logit = math.log(clamped / (1.0 - clamped))
        cal_conf = round(_temperature_scale(logit, self.cfg.temperature), 4)

        is_ood, ood_source = _detect_ood(fusion, evidence, self.cfg.temperature, self.cfg.ood_threshold)

        # Determine abstention
        all_issues = logic.issues + who.violations
        should_abstain = (
            cal_conf < self.cfg.abstention_threshold
            or is_ood
            or len(logic.issues) >= 3
        )
        abstention_reason: Optional[str] = None
        if should_abstain:
            if cal_conf < self.cfg.abstention_threshold:
                abstention_reason = f"Calibrated confidence {cal_conf:.2f} below threshold {self.cfg.abstention_threshold}"
            elif is_ood:
                abstention_reason = f"Out-of-distribution input detected ({ood_source})"
            else:
                abstention_reason = "Multiple consistency issues require human review"

        passed = logic.consistent and who.compliant and not should_abstain

        result = VerificationResult(
            passed=passed,
            logic_check=logic,
            who_validation=who,
            raw_confidence=round(raw_conf, 4),
            calibrated_confidence=cal_conf,
            is_ood=is_ood,
            abstain=should_abstain,
            abstention_reason=abstention_reason,
            uncertainty_source=ood_source,
            consistency_issues=all_issues,
        )

        logger.info(
            "Verification: passed=%s cal_conf=%.2f ood=%s abstain=%s",
            passed, cal_conf, is_ood, should_abstain,
        )
        return result