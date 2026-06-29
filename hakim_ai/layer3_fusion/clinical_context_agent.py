"""
Layer 3 — Clinical Context Agent.

Encodes structured EHR data, clinical history, and prior molecular
test results into a natural-language prompt suitable for LLM fusion.

Real implementation: parse FHIR resources / HL7 messages or read from
the institutional LIMS; encode with a clinical NLP model or structure
as a system prompt for the reasoning LLM.
"""
from __future__ import annotations

from typing import Any, List, Optional

from hakim_ai.types import ClinicalContext, ClinicalInput, EvidenceBundle
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer3.clinical_context")

_RISK_FACTOR_MAP = {
    "h_pylori": "H. pylori infection history",
    "family_history": "family history of gastric cancer",
    "prior_gastrectomy": "prior gastrectomy",
    "intestinal_metaplasia": "known intestinal metaplasia",
    "smoking": "smoking history",
    "high_salt_diet": "high-salt diet",
}


class ClinicalContextAgent:
    """
    Encodes clinical input into a structured text prompt for downstream agents.

    Inputs:  Optional[ClinicalInput], Optional[EvidenceBundle]
    Outputs: ClinicalContext
    """

    def __init__(self, encoder: Optional[Any] = None):
        self.encoder = encoder

    def run(self, clinical_input: Optional[ClinicalInput], evidence: Optional[EvidenceBundle] = None) -> ClinicalContext:
        if clinical_input is None:
            logger.warning("No clinical data — using image-only context")
            return self._empty_context()

        logger.info(
            "Clinical context encoding for patient %s", clinical_input.patient_id
        )

        history_items = self._extract_history(clinical_input)
        risk_factors = self._identify_risk_factors(clinical_input)
        prior_results = self._parse_prior_molecular(clinical_input)
        prompt = self._build_prompt(clinical_input, history_items, risk_factors)

        context = ClinicalContext(
            encoded_prompt=prompt,
            relevant_history_items=history_items,
            prior_molecular_results=prior_results,
            risk_factors=risk_factors,
            cross_modal_relevance=self._compute_cross_modal_attention(prompt, evidence)
        )
        logger.debug("Clinical prompt length: %d chars", len(prompt))
        return context

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_history(self, ci: ClinicalInput) -> List[str]:
        items: List[str] = []
        if ci.age:
            items.append(f"Age: {ci.age} years")
        if ci.sex:
            items.append(f"Sex: {ci.sex}")
        if ci.biopsy_location:
            items.append(f"Biopsy site: {ci.biopsy_location}")
        if ci.endoscopy_findings:
            items.append(f"Endoscopy: {ci.endoscopy_findings}")
        if ci.prior_treatments:
            items.append(f"Prior treatments: {', '.join(ci.prior_treatments)}")
        if ci.ehr_notes:
            # Truncate long notes
            notes = ci.ehr_notes[:500] + "..." if len(ci.ehr_notes) > 500 else ci.ehr_notes
            items.append(f"Clinical notes: {notes}")
        return items

    def _identify_risk_factors(self, ci: ClinicalInput) -> List[str]:
        factors: List[str] = []
        if ci.h_pylori_status:
            factors.append("H. pylori infection history")
        if ci.family_history:
            factors.append(f"Family history: {ci.family_history}")
        if ci.prior_treatments and any(
            "chemo" in t.lower() for t in ci.prior_treatments
        ):
            factors.append("Prior chemotherapy")
        return factors

    def _parse_prior_molecular(self, ci: ClinicalInput) -> dict:
        """Extract prior molecular test results from free-text reports."""
        if not ci.molecular_reports:
            return {}
        results: dict = {}
        text = ci.molecular_reports.lower()
        if "msi-h" in text or "mismatch repair deficient" in text:
            results["MSI"] = "MSI-H (prior report)"
        elif "mss" in text or "mismatch repair proficient" in text:
            results["MSI"] = "MSS (prior report)"
        if "her2 positive" in text or "her2 3+" in text:
            results["HER2"] = "positive (prior report)"
        elif "her2 negative" in text:
            results["HER2"] = "negative (prior report)"
        return results

    def _build_prompt(
        self,
        ci: ClinicalInput,
        history: List[str],
        risk_factors: List[str],
    ) -> str:
        lines = [
            f"Patient ID: {ci.patient_id}",
            "",
            "Clinical History:",
            *[f"  - {item}" for item in history],
        ]
        if risk_factors:
            lines += ["", "Risk Factors:", *[f"  - {rf}" for rf in risk_factors]]
        return "\n".join(lines)

    def _empty_context(self) -> ClinicalContext:
        return ClinicalContext(
            encoded_prompt="No clinical data available. Analysis based on histology only.",
            relevant_history_items=[],
            prior_molecular_results={},
            risk_factors=[],
            cross_modal_relevance=0.0,
        )

    def _compute_cross_modal_attention(self, prompt: str, evidence: Optional[EvidenceBundle]) -> float:
        """
        Compute cross-modal attention/relevance score between clinical text and H&E features.
        Real implementation: use CONCH text encoder and compute cosine similarity with mean patch vector.
        """
        if evidence is None or not evidence.navigation.selected_patches:
            return 0.5
            
        # Collect mean patch feature vector
        features = [p.feature_vector for p in evidence.navigation.selected_patches if getattr(p, "feature_vector", None) is not None]
        if not features:
            return 0.5
            
        import numpy as np
        try:
            mean_visual_feature = np.mean(features, axis=0)
            
            if self.encoder and hasattr(self.encoder, "encode_text"):
                # Real cross-modal dot product attention
                text_feature = np.array(self.encoder.encode_text(prompt))
                
                # Normalize both (they should already be, but just in case)
                mean_v_norm = np.linalg.norm(mean_visual_feature)
                text_norm = np.linalg.norm(text_feature)
                
                if mean_v_norm > 0 and text_norm > 0:
                    sim = np.dot(mean_visual_feature, text_feature) / (mean_v_norm * text_norm)
                    # Scale from [-1, 1] to [0, 1]
                    relevance = float((sim + 1.0) / 2.0)
                    return round(relevance, 4)

            # Fallback mock logic if encoder is missing or it's a mock
            desc_text = " ".join([d.narrative.lower() for d in evidence.descriptions])
            prompt_lower = prompt.lower()
            
            # Simple keyword matching as a proxy for attention
            overlap = 0
            if "msi" in prompt_lower and "lymphocyte" in desc_text:
                overlap += 1
            if "poorly differentiated" in prompt_lower and "poorly cohesive" in desc_text:
                overlap += 1
            if "h. pylori" in prompt_lower and "intestinal" in desc_text:
                overlap += 1
                
            relevance = 0.5 + min(overlap * 0.15, 0.45)
            return round(relevance, 4)
        except Exception as e:
            logger.warning(f"Cross-modal attention failed: {e}")
            return 0.5