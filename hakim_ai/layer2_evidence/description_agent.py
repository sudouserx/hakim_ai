"""
Layer 2 — Description Agent.

Generates clinician-quality natural language descriptions for the top
diagnostic patches selected by the Navigation Agent.

This is the primary explainability mechanism, following the PathFinder
(ICCV 2025) design pattern: NL descriptions are more interpretable to
pathologists than raw attention maps.

Testing: uses PathChatVLM in templated descriptions mode.
Real: forward selected patches through PathChat or CONCH+GPT-4 Vision.
"""
from __future__ import annotations

from typing import Any, List

from hakim_ai.config import DescriptionConfig
from hakim_ai.foundation_models.base_encoder import BaseVLM
from hakim_ai.types import (
    NavigationResult,
    PatchDescription,
    WSIData,
)
from hakim_ai.utils.image_utils import extract_patch_from_wsi
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer2.description")

_MORPHOLOGICAL_FEATURES = [
    # Drawn from WHO gastric cancer diagnostic criteria
    "irregular glandular architecture",
    "loss of normal mucosal organisation",
    "enlarged nuclei with prominent nucleoli",
    "increased nuclear-to-cytoplasmic ratio",
    "signet-ring cell morphology",
    "poorly cohesive tumour cells",
    "desmoplastic stromal reaction",
    "tumour-infiltrating lymphocytes (TILs)",
    "mucinous component",
    "atypical mitotic figures",
    "perineural invasion",
    "lymphovascular invasion",
    "goblet cells (intestinal metaplasia)",
    "nuclear pleomorphism",
    "necrotic debris within glands",
]

_PROMPTS = {
    "describe": (
        "Describe the histomorphological features of this H&E stained patch "
        "from a gastric biopsy. Note glandular architecture, nuclear features, "
        "stromal composition, and any immune infiltrate."
    ),
    "msi": (
        "Are there features in this patch suggestive of microsatellite instability "
        "(e.g. high TIL density, medullary growth pattern, mucinous differentiation)?"
    ),
    "lauren": (
        "Does the growth pattern in this patch suggest Lauren intestinal type "
        "(gland-forming) or diffuse type (poorly cohesive cells)?"
    ),
}


class DescriptionAgent:
    """
    Natural-language patch description agent.

    Inputs:  WSIData, NavigationResult
    Outputs: List[PatchDescription]
    """

    def __init__(self, cfg: DescriptionConfig, vlm: BaseVLM, normalizer: Any = None):
        self.cfg = cfg
        self.vlm = vlm
        self.normalizer = normalizer

    def run(
        self, wsi_data: WSIData, navigation: NavigationResult
    ) -> List[PatchDescription]:
        logger.info(
            "Description agent started for patient %s — describing top %d patches",
            wsi_data.patient_id,
            self.cfg.max_patches_to_describe,
        )

        top_patches = navigation.selected_patches[: self.cfg.max_patches_to_describe]
        descriptions: List[PatchDescription] = []

        for patch_coord in top_patches:
            # Extract the patch image from WSI at these coords
            patch_image = extract_patch_from_wsi(
                wsi_path=wsi_data.wsi_path,
                x=patch_coord.x,
                y=patch_coord.y,
                level=patch_coord.level,
                size=(patch_coord.width, patch_coord.height),
                slide_handle=getattr(wsi_data, "slide_handle", None),
                normalizer=self.normalizer,
            )

            narrative = self.vlm.describe_patch(
                patch=patch_image,
                prompt=_PROMPTS["describe"],
            )
            features = self._extract_morphological_features(narrative)
            confidence = self._estimate_description_confidence(
                patch_coord.importance_score
            )

            descriptions.append(
                PatchDescription(
                    patch_coord=patch_coord,
                    narrative=narrative,
                    morphological_features=features,
                    confidence=confidence,
                    magnification=self._level_to_mag(patch_coord.level),
                )
            )

        logger.info(
            "Description done: %d patches described", len(descriptions)
        )
        return descriptions

    def describe_for_question(
        self, wsi_data: WSIData, navigation: NavigationResult, question_key: str
    ) -> List[str]:
        """
        Run a targeted question (e.g. 'msi', 'lauren') over top patches.
        Used by the interactive Q&A interface.
        """
        prompt = _PROMPTS.get(question_key, question_key)
        answers = []
        for patch_coord in navigation.selected_patches[:3]:
            patch_image = extract_patch_from_wsi(
                wsi_path=wsi_data.wsi_path,
                x=patch_coord.x,
                y=patch_coord.y,
                level=patch_coord.level,
                size=(patch_coord.width, patch_coord.height),
                slide_handle=getattr(wsi_data, "slide_handle", None),
                normalizer=self.normalizer,
            )
            answer = self.vlm.answer_question(patch=patch_image, question=prompt)
            answers.append(answer)
        return answers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_morphological_features(self, narrative: str) -> List[str]:
        """Extract morphological keywords from the narrative text."""
        found = []
        lower = narrative.lower()
        for feat in _MORPHOLOGICAL_FEATURES:
            if any(word in lower for word in feat.split()[:2]):
                found.append(feat)
        # Ensure at least 2 features
        if len(found) < 2:
            found = _MORPHOLOGICAL_FEATURES[:3]
        return found[:6]

    def _estimate_description_confidence(self, importance_score: float) -> float:
        """Higher importance patch → higher description confidence."""
        return round(0.6 + importance_score * 0.35, 3)

    def _level_to_mag(self, level: int) -> float:
        return {0: 40.0, 1: 20.0, 2: 10.0, 3: 5.0}.get(level, 20.0)