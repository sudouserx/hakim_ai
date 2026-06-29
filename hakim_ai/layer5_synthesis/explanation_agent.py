"""
Layer 5 — Explanation Agent.

Generates structured, clinician-facing natural language explanations
linking the AI's evidence to the final diagnosis. Follows the
PathFinder (ICCV 2025) design: NL descriptions grounded in specific
patch evidence, not generic heatmap summaries.

Key requirements from the architecture document:
  - Concept-aligned: mapped to WHO classification criteria
  - Uncertainty-aware: explicitly states what is certain vs. uncertain
  - Actionable abstention: flags what additional tests would resolve uncertainty
  - Non-over-explaining: avoids information overload that causes over-reliance
"""
from __future__ import annotations

from typing import List, Optional

from hakim_ai.types import (
    Diagnosis,
    Explanation,
    EvidenceBundle,
    FusionResult,
    MSIStatus,
    VerificationResult,
)
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer5.explanation")


class ExplanationAgent:
    """
    Natural language explanation generator.

    Inputs:  Diagnosis, EvidenceBundle, FusionResult, VerificationResult
    Outputs: Explanation
    """

    def run(
        self,
        diagnosis: Diagnosis,
        evidence: EvidenceBundle,
        fusion: FusionResult,
        verification: VerificationResult,
    ) -> Explanation:
        logger.info("Explanation generation started")

        narrative = self._build_narrative(diagnosis, evidence, fusion, verification)
        citations = self._build_citations(evidence, fusion)
        concept_alignments = self._align_to_who(fusion, verification)
        uncertainty_stmt = self._build_uncertainty_statement(verification, fusion)
        key_features = self._extract_key_features(evidence)
        counterfactual = self._build_counterfactual(fusion, evidence)

        explanation = Explanation(
            narrative=narrative,
            evidence_citations=citations,
            concept_alignments=concept_alignments,
            uncertainty_statement=uncertainty_stmt,
            key_morphological_features=key_features,
            counterfactual_note=counterfactual,
        )

        logger.info("Explanation generated (%d chars narrative)", len(narrative))
        return explanation

    # ------------------------------------------------------------------
    # Narrative construction
    # ------------------------------------------------------------------

    def _build_narrative(
        self,
        diagnosis: Diagnosis,
        evidence: EvidenceBundle,
        fusion: FusionResult,
        verification: VerificationResult,
    ) -> str:
        mol = fusion.molecular
        seg = evidence.segmentation
        sections: List[str] = []

        # 1. Primary diagnosis statement
        sections.append(
            f"The histopathological examination is consistent with {diagnosis.primary_diagnosis}. "
            f"The overall diagnostic confidence is {verification.calibrated_confidence:.0%}."
        )

        # 2. Morphological evidence (from patch descriptions)
        if evidence.descriptions:
            top_desc = evidence.descriptions[0]
            sections.append(
                f"The most diagnostically significant region (importance score "
                f"{top_desc.patch_coord.importance_score:.2f} at "
                f"{top_desc.magnification:.0f}× magnification) shows: "
                f"{top_desc.narrative}"
            )
            if len(evidence.descriptions) > 1:
                sections.append(
                    f"Supporting regions demonstrate: "
                    + "; ".join(
                        d.narrative[:120] + "..." for d in evidence.descriptions[1:3]
                    )
                )

        # 3. Tissue composition
        sections.append(
            f"Tissue compartment analysis: tumour {seg.tumour_fraction:.0%}, "
            f"stroma {seg.stroma_fraction:.0%}, "
            f"TIL density {seg.til_density:.0%}, "
            f"necrosis {seg.necrosis_fraction:.0%}."
        )

        # 4. Molecular prediction rationale
        msi_rationale = (
            f"High TIL density ({seg.til_density:.0%}) supports the "
            f"MSI-H prediction (P={mol.msi_probability:.2f}). "
            f"MMR immunohistochemistry (MLH1/MSH2/MSH6/PMS2) is recommended for confirmation."
            if mol.msi_status == MSIStatus.MSI_HIGH
            else f"TIL density ({seg.til_density:.0%}) and tumour morphology "
            f"are more consistent with microsatellite-stable (MSS) disease "
            f"(P(MSI-H)={mol.msi_probability:.2f})."
        )
        sections.append(msi_rationale)

        # 5. HER2 comment
        sections.append(
            f"HER2 approximation from H&E morphology: {mol.her2_status.value} "
            f"(P={mol.her2_probability:.2f}). "
            f"Formal HER2 IHC/FISH scoring is required for treatment eligibility determination."
        )

        # 6. Consistency check summary
        if verification.consistency_issues:
            issues_text = " | ".join(verification.consistency_issues[:2])
            sections.append(
                f"⚠ Consistency notes requiring attention: {issues_text}"
            )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Citations
    # ------------------------------------------------------------------

    def _build_citations(
        self, evidence: EvidenceBundle, fusion: FusionResult
    ) -> List[str]:
        citations: List[str] = []

        for i, desc in enumerate(evidence.descriptions[:3]):
            coord = desc.patch_coord
            citations.append(
                f"[Patch {i+1}] Coordinates (x={coord.x}, y={coord.y}) "
                f"at level {coord.level} ({desc.magnification:.0f}×), "
                f"importance={coord.importance_score:.2f}: \"{desc.narrative[:80]}...\""
            )

        # Knowledge base citations
        for passage in fusion.knowledge.literature_evidence[:2]:
            citations.append(f"[Literature] {passage[:120]}...")

        return citations

    # ------------------------------------------------------------------
    # WHO concept alignment
    # ------------------------------------------------------------------

    def _align_to_who(
        self, fusion: FusionResult, verification: VerificationResult
    ) -> List[str]:
        alignments: List[str] = []
        who = verification.who_validation
        if who.suggested_classification:
            alignments.append(
                f"WHO 5th Edition classification: {who.suggested_classification}"
            )
        alignments.extend(f"Matched criterion: {c[:80]}" for c in who.matched_criteria[:2])
        if fusion.knowledge.who_criteria:
            alignments.append(
                f"Retrieved guideline: {fusion.knowledge.who_criteria[0][:100]}..."
            )
        return alignments

    # ------------------------------------------------------------------
    # Uncertainty statement
    # ------------------------------------------------------------------

    def _build_uncertainty_statement(
        self, verification: VerificationResult, fusion: FusionResult
    ) -> str:
        conf = verification.calibrated_confidence
        mol = fusion.molecular

        certain_parts: List[str] = []
        uncertain_parts: List[str] = []

        # Lauren type
        if mol.lauren_confidence > 0.75:
            certain_parts.append(f"Lauren type ({mol.lauren_class.value})")
        else:
            uncertain_parts.append(f"Lauren type (confidence {mol.lauren_confidence:.0%})")

        # MSI
        msi_conf = max(mol.msi_probability, 1 - mol.msi_probability)
        if msi_conf > 0.75:
            certain_parts.append("MSI status (from H&E features)")
        else:
            uncertain_parts.append(f"MSI status (H&E confidence {msi_conf:.0%} — IHC required)")

        # HER2 always uncertain from H&E
        uncertain_parts.append("HER2 status (IHC/FISH always required)")

        parts: List[str] = []
        if certain_parts:
            parts.append(f"Confident findings: {', '.join(certain_parts)}.")
        if uncertain_parts:
            parts.append(
                f"Uncertain — recommend confirmation: {', '.join(uncertain_parts)}."
            )
        if verification.is_ood:
            parts.append(
                f"⚠ OOD flag: {verification.uncertainty_source} — "
                f"this case may be outside the model's training distribution."
            )
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Counterfactual and key features
    # ------------------------------------------------------------------

    def _extract_key_features(self, evidence: EvidenceBundle) -> List[str]:
        features: set = set()
        for desc in evidence.descriptions[:3]:
            features.update(desc.morphological_features[:3])
        return list(features)[:8]

    def _build_counterfactual(
        self, fusion: FusionResult, evidence: EvidenceBundle
    ) -> Optional[str]:
        """
        Provide a contrastive 'what would change the prediction' statement.
        Per architecture doc: "What would change the prediction?" is clinically intuitive.
        Full counterfactual at WSI scale is technically unsolved — we provide a
        verbal approximation.
        """
        mol = fusion.molecular
        if mol.msi_probability > 0.5 and evidence.segmentation.til_density > 0.1:
            return (
                "If TIL density were <5% and tumour growth pattern were predominantly "
                "glandular without necrosis, the MSI-H prediction would likely shift "
                "toward MSS. Conversely, a medullary growth pattern would further "
                "support MSI-H classification."
            )
        return (
            "If glandular differentiation were more prominent and TIL density lower, "
            "a higher Lauren intestinal confidence and lower MSI-H probability would "
            "be expected."
        )