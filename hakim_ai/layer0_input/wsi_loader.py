"""
Layer 0 — WSI and clinical data loaders.

OpenSlideWSILoader: real implementation using openslide-python.
Install: pip install openslide-python (also requires the C library).
"""
from __future__ import annotations

import abc
import os
import random
from typing import Any, Dict, List, Optional

from hakim_ai.types import WSIInput, WSIData, ClinicalInput
from hakim_ai.utils.logging_utils import get_logger

logger = get_logger("layer0.loader")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseWSILoader(abc.ABC):
    @abc.abstractmethod
    def load(self, wsi_input: WSIInput) -> WSIData:
        """Load a WSI and return a WSIData object."""
        ...


# ---------------------------------------------------------------------------
# Loader without dependencies
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# OpenSlide loader (real; requires openslide-python)
# ---------------------------------------------------------------------------

class OpenSlideWSILoader(BaseWSILoader):
    """
    Real WSI loader using openslide-python.

    Requires system dependency `openslide-tools` (e.g., `apt-get install openslide-tools`).
    Reads level dimensions, extracts a thumbnail, and records tile metadata.
    Does NOT load all tiles into memory — downstream agents extract patches
    on demand via patch coordinates.
    """

    THUMBNAIL_SIZE = (512, 512)

    def load(self, wsi_input: WSIInput) -> WSIData:
        try:
            import openslide
        except ImportError as exc:
            raise ImportError(
                "openslide-python is required for real WSI loading. "
                "Install: pip install openslide-python"
            ) from exc

        if not os.path.exists(wsi_input.wsi_path):
            raise FileNotFoundError(f"WSI not found: {wsi_input.wsi_path}")

        # Basic format check
        valid_extensions = {".svs", ".ndpi", ".tiff", ".tif", ".mrxs", ".vsi", ".bif"}
        if not any(wsi_input.wsi_path.lower().endswith(ext) for ext in valid_extensions):
            logger.warning(f"File {wsi_input.wsi_path} does not have a typical WSI extension.")

        slide = openslide.OpenSlide(wsi_input.wsi_path)
        thumb = slide.get_thumbnail(self.THUMBNAIL_SIZE)
        import numpy as np
        thumb_array = np.array(thumb)

        dimensions = [slide.level_dimensions[i] for i in range(slide.level_count)]
        mpp_x = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
        if mpp_x <= 0.0:
            logger.warning("MPP reported as <= 0.0, falling back to 0.25")
            mpp_x = 0.25
            
        metadata = dict(slide.properties)

        # We keep the slide handle open and store it in WSIData for downstream use
        return WSIData(
            patient_id=wsi_input.patient_id,
            wsi_path=wsi_input.wsi_path,
            thumbnail=thumb_array,
            tile_paths=[],
            level_dimensions=dimensions,
            level_count=len(dimensions),
            mpp=mpp_x,
            metadata=metadata,
            slide_handle=slide,
        )


# ---------------------------------------------------------------------------
# Clinical data loader
# ---------------------------------------------------------------------------

class ClinicalLoader:
    """
    Loads and validates clinical / EHR data.
    Real implementation: query FHIR endpoint or parse HL7 / CSV export.
    """

    def load(self, clinical_input: Optional[ClinicalInput]) -> Optional[ClinicalInput]:
        if clinical_input is None:
            logger.warning("No clinical input provided — running image-only mode")
            return None
        logger.debug("Clinical data loaded for patient %s", clinical_input.patient_id)
        return clinical_input

    def from_dict(self, d: Dict[str, Any], patient_id: str) -> ClinicalInput:
        return ClinicalInput(
            patient_id=patient_id,
            age=d.get("age"),
            sex=d.get("sex"),
            ehr_notes=d.get("ehr_notes"),
            molecular_reports=d.get("molecular_reports"),
            endoscopy_findings=d.get("endoscopy_findings"),
            prior_treatments=d.get("prior_treatments", []),
            h_pylori_status=d.get("h_pylori_status"),
            family_history=d.get("family_history"),
            biopsy_location=d.get("biopsy_location"),
        )


def build_wsi_loader() -> BaseWSILoader:
    """Factory: return real OpenSlide loader."""
    return OpenSlideWSILoader()