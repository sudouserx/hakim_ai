"""Foundation model adapters package."""
from hakim_ai.foundation_models.base_encoder import BaseEncoder, BaseVLM
from hakim_ai.foundation_models.uni_adapter import UNI2Encoder
from hakim_ai.foundation_models.conch_adapter import CONCHEncoder, PathChatVLM
from hakim_ai.config import FoundationModelConfig


def build_patch_encoder(cfg: FoundationModelConfig) -> BaseEncoder:
    """Factory: return the configured patch encoder."""
    name = cfg.patch_encoder.lower()
    mock = cfg.mock_mode
    if name in ("uni2", "uni"):
        return UNI2Encoder(mock_mode=mock)
    if name in ("conch", "titan"):
        return CONCHEncoder(mock_mode=mock)
    raise ValueError(f"Unknown patch encoder: {cfg.patch_encoder}")


def build_vlm(cfg: FoundationModelConfig) -> BaseVLM:
    """Factory: return the configured VLM."""
    name = cfg.vlm.lower()
    mock = cfg.mock_mode
    if name == "pathchat":
        return PathChatVLM(mock_mode=mock)
    raise ValueError(f"Unknown VLM: {cfg.vlm}")


__all__ = [
    "BaseEncoder",
    "BaseVLM",
    "UNI2Encoder",
    "CONCHEncoder",
    "PathChatVLM",
    "build_patch_encoder",
    "build_vlm",
]