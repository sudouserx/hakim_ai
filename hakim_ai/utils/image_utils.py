"""
Image utilities for the histopathology pipeline.

All functions that touch real pixel data are clearly separated from test
versions so real implementations can be swapped in without changing callers.
"""
from __future__ import annotations

import math
import random
import functools
from typing import List, Optional, Tuple, Any

import numpy as np
from PIL import Image

try:
    import openslide
except ImportError:
    openslide = None

# ---------------------------------------------------------------------------
# Stain normalization
# ---------------------------------------------------------------------------

class StainNormalizerMacenko:
    """
    Macenko stain normalization implementation.

    Real implementation using numpy for SVD-based stain matrix estimation.
    """

    def __init__(self, beta: float = 0.15, alpha: float = 1.0):
        self.beta = beta
        self.alpha = alpha
        self.HERef = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def normalize(self, wsi_path: str) -> str:
        """
        Normalise a WSI and return the path to the normalised file.
        Real normalization of whole WSIs is out of scope for this pipeline 
        (usually done patch-by-patch). Returning original path.
        """
        return wsi_path

    def normalize_patch(self, patch: Any) -> Any:
        """Normalises a patch (PIL Image or nested list) in-place/returned."""
        if isinstance(patch, list):
            # It's a list, return as is
            return patch
        
        img = np.array(patch)
        if img.shape[2] == 4:
            img = img[:, :, :3] # RGBA to RGB
            
        h, w, c = img.shape
        img_reshaped = img.reshape((-1, 3))
        
        # Convert RGB to OD
        OD = -np.log10((img_reshaped.astype(np.float32) + 1) / 256.0)
        
        # Remove data with OD intensity less than beta
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        
        if len(ODhat) == 0:
            return patch
            
        # SVD on OD
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        
        # Project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = ODhat.dot(eigvecs[:, 1:3])
        
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)
        
        vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
        vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)
        
        # Ensure H stain is first
        if vMin[0] > vMax[0]:
            HE = np.array((vMin[:, 0], vMax[:, 0])).T
        else:
            HE = np.array((vMax[:, 0], vMin[:, 0])).T
            
        # Determine concentrations
        Y = np.reshape(OD, (-1, 3)).T
        try:
            C = np.linalg.lstsq(HE, Y, rcond=None)[0]
        except Exception:
            return patch
            
        # Normalize
        maxC = np.percentile(C, 99, axis=1)
        # Avoid division by zero
        maxC[maxC == 0] = 1.0
        
        C_norm = np.dot(np.diag(self.maxCRef / maxC), C)
        
        # Reconstruct
        OD_norm = np.dot(self.HERef, C_norm)
        img_norm = 255 * np.exp(-1 * OD_norm)
        img_norm = np.clip(img_norm, 0, 255).astype(np.uint8).T.reshape((h, w, 3))
        
        return Image.fromarray(img_norm)


class StainNormalizerReinhard:
    """Reinhard colour transfer implementation."""
    
    def normalize(self, wsi_path: str) -> str:
        return wsi_path

    def normalize_patch(self, patch: Any) -> Any:
        if isinstance(patch, list):
            return patch
        
        # Minimal Reinhard requires a reference image stats.
        # We skip full implementation for now and just return the image
        return patch


class PassthroughNormalizer:
    """No-op normalizer for pre-normalised data."""

    def normalize(self, wsi_path: str) -> str:
        return wsi_path

    def normalize_patch(self, patch_array: Any) -> Any:
        return patch_array


def build_normalizer(method: str):
    """Factory function for stain normalizers."""
    m = method.lower()
    if m == "macenko":
        return StainNormalizerMacenko()
    if m == "reinhard":
        return StainNormalizerReinhard()
    if m in ("passthrough", "none"):
        return PassthroughNormalizer()
    raise ValueError(f"Unknown stain normalizer: {method}")


# ---------------------------------------------------------------------------
# Tile / patch extraction
# ---------------------------------------------------------------------------

def compute_tissue_mask(
    thumbnail: Optional[Any],
    tissue_threshold: float = 0.8,
) -> List[List[bool]]:
    """
    Return a binary mask of tissue vs. background on a thumbnail.
    Real implementation: Otsu thresholding on saturation channel.
    """
    if thumbnail is None:
        raise ValueError("Thumbnail is required to compute tissue mask.")
    
    # Real image (PIL or numpy)
    img = np.array(thumbnail)
    
    if len(img.shape) == 3 and img.shape[2] >= 3:
        # Convert RGB to HSV
        import matplotlib.colors as colors
        hsv = colors.rgb_to_hsv(img[:, :, :3] / 255.0)
        saturation = hsv[:, :, 1]
    else:
        saturation = img
        
    # Otsu thresholding on saturation
    from skimage.filters import threshold_otsu
    try:
        thresh = threshold_otsu(saturation)
        mask = saturation > thresh
    except Exception:
        # Fallback if image has single value
        mask = saturation > np.mean(saturation)
        
    return mask.tolist()


def tissue_coverage(mask: List[List[bool]]) -> float:
    """Return fraction of mask pixels marked as tissue."""
    if not mask:
        return 0.0
    total = sum(len(row) for row in mask)
    tissue = sum(1 for row in mask for v in row if v)
    return tissue / max(total, 1)


def extract_patch_coordinates(
    tissue_mask: List[List[bool]],
    patch_size: int,
    thumbnail_downsample: int,
    top_k: int = 20,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """
    Sample patch coordinates from tissue-positive regions.
    Returns (x, y) coordinates at full WSI resolution.
    
    Real implementation: grid-based sampling with tissue filtering.
    """
    # Create systematic grid points
    grid_coords = []
    
    # Convert mask to numpy for easier handling
    mask_np = np.array(tissue_mask)
    if mask_np.size == 0:
        return []
        
    rows, cols = mask_np.shape
    
    # Downsample factor mapping from full res to thumbnail
    # So full_x = col * downsample
    for r in range(0, rows):
        for c in range(0, cols):
            if mask_np[r, c]:
                grid_coords.append((int(c * thumbnail_downsample), int(r * thumbnail_downsample)))

    rng = random.Random(seed)
    # Return up to top_k random valid coordinates
    k = min(top_k, len(grid_coords))
    return rng.sample(grid_coords, k) if grid_coords else []


def extract_patch_from_wsi(
    wsi_path: str, x: int, y: int, level: int, size: Tuple[int, int] = (512, 512),
    slide_handle: Any = None, normalizer: Any = None
) -> Any:
    """
    Extract a patch from the WSI using openslide.

    Args:
        normalizer: Optional stain normalizer instance (e.g. StainNormalizerMacenko).
                    If provided, the extracted patch is stain-normalized before return.
    """
    patch = None
    if slide_handle is None:
        if openslide is None:
            raise RuntimeError("openslide-python is required to extract patches.")

        # Real extraction without handle (opens and closes)
        try:
            with openslide.OpenSlide(wsi_path) as slide:
                patch = slide.read_region((x, y), level, size).convert("RGB")
        except Exception:
            return None
    else:
        # Use provided handle
        try:
            patch = slide_handle.read_region((x, y), level, size).convert("RGB")
        except Exception:
            return None

    # Apply stain normalization if a normalizer is provided
    if patch is not None and normalizer is not None:
        try:
            patch = normalizer.normalize_patch(patch)
        except Exception:
            pass  # Return un-normalized patch rather than failing

    return patch


# ---------------------------------------------------------------------------
# Colour / quality metrics
# ---------------------------------------------------------------------------

def estimate_focus_quality(patch_array: Optional[Any]) -> float:
    """
    Estimate focus quality using Laplacian variance.
    Real implementation: compute variance of the Laplacian of the grayscale patch.
    """
    if patch_array is None:
        return 0.0

    img = np.array(patch_array)
    if len(img.shape) == 3:
        # RGB to Gray
        gray = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])
    else:
        gray = img
        
    import scipy.ndimage as ndimage
    lap = ndimage.laplace(gray)
    var = np.var(lap)
    
    # Normalize to 0-1 range (heuristic)
    score = min(1.0, var / 500.0)
    return round(max(0.0, score), 3)


def estimate_stain_quality(patch_array: Optional[Any]) -> float:
    """
    Estimate H&E stain quality from OD histogram entropy.
    """
    if patch_array is None:
        return 0.0
        
    img = np.array(patch_array)
    if len(img.shape) < 3 or img.shape[2] < 3:
        return 0.5
        
    # Convert to OD
    img_reshaped = img[:, :, :3].reshape((-1, 3))
    OD = -np.log10((img_reshaped.astype(np.float32) + 1) / 256.0)
    
    # Compute histogram entropy of OD sum
    od_sum = np.sum(OD, axis=1)
    hist, _ = np.histogram(od_sum, bins=256, range=(0, 3), density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))
    
    # Normalize (max entropy for 256 bins is ~5.5)
    score = min(1.0, entropy / 4.5)
    return round(score, 3)


def detect_artifacts(patch_array: Optional[Any]) -> List[str]:
    """
    Detect common slide artifacts: tissue folds, pen marks, out-of-focus.
    """
    if patch_array is None:
        return []
        
    artifacts = []
    
    # Check focus
    focus = estimate_focus_quality(patch_array)
    if focus < 0.2:
        artifacts.append("out_of_focus")
        
    img = np.array(patch_array)
    
    # Pen marks detection (highly saturated non-pink/purple colors)
    if len(img.shape) == 3 and img.shape[2] >= 3:
        import matplotlib.colors as colors
        hsv = colors.rgb_to_hsv(img[:, :, :3] / 255.0)
        # Find pixels with high saturation but hue outside pink/purple range
        # Pink/purple H&E hue is roughly around 0.8-1.0 and 0.0-0.1
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        
        pen_mask = (sat > 0.8) & ((hue > 0.15) & (hue < 0.75))
        if np.mean(pen_mask) > 0.01:  # More than 1% is pen mark
            artifacts.append("pen_mark")
            
        # Tissue folds: look for very dark regions in OD
        img_reshaped = img[:, :, :3].reshape((-1, 3))
        OD = -np.log10((img_reshaped.astype(np.float32) + 1) / 256.0)
        od_sum = np.sum(OD, axis=1)
        if np.mean(od_sum > 2.0) > 0.05:  # High optical density = tissue fold
            artifacts.append("tissue_fold")
            
    return artifacts