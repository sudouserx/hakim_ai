"""
Abstract base class for all foundation model adapters.

Pattern: every real model (UNI 2, CONCH, PathChat, Virchow 2) implements
BaseEncoder / BaseVLM. Test adapters return deterministic synthetic vectors
so the pipeline runs end-to-end without GPU or model weights.
"""
from __future__ import annotations

import abc
import hashlib
import math
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Base encoder (patch / slide feature extraction)
# ---------------------------------------------------------------------------

class BaseEncoder(abc.ABC):
    """Abstract patch or slide feature encoder."""

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int: ...

    def load(self) -> None:
        """Load the foundation model weights into memory."""
        pass

    def unload(self) -> None:
        """Unload the foundation model weights to free up memory."""
        pass

    def encode_patch(self, patch: Any) -> List[float]:
        """Return a 1-D feature vector for a single patch (PIL Image or np.ndarray)."""
        feat = self._encode_patch(patch)
        self._validate_feature(feat)
        return feat

    def encode_batch(self, patches: List[Any]) -> List[List[float]]:
        """Return a list of feature vectors for a batch of patches."""
        feats = self._encode_batch(patches)
        for f in feats:
            self._validate_feature(f)
        return feats
        
    def _validate_feature(self, feat: Any) -> None:
        """Post-condition validation to prevent test/real drift."""
        if not isinstance(feat, list):
            raise TypeError(f"Encoder must return a list of floats, got {type(feat)}")
        if len(feat) != self.embedding_dim:
            raise ValueError(f"Encoder returned dimension {len(feat)}, expected {self.embedding_dim}")
        if len(feat) > 0 and not isinstance(feat[0], float):
            raise TypeError(f"Encoder elements must be float, got {type(feat[0])}")

    @abc.abstractmethod
    def _encode_patch(self, patch: Any) -> List[float]:
        ...

    @abc.abstractmethod
    def _encode_batch(self, patches: List[Any]) -> List[List[float]]:
        ...


class BaseVLM(abc.ABC):
    """Abstract vision-language model for patch description."""

    def load(self) -> None:
        """Load the foundation model weights into memory."""
        pass

    def unload(self) -> None:
        """Unload the foundation model weights to free up memory."""
        pass

    @abc.abstractmethod
    def describe_patch(self, patch: Any, prompt: str = "") -> str:
        """Return a natural-language description of a histopathology patch."""
        ...

    @abc.abstractmethod
    def answer_question(self, patch: Any, question: str) -> str:
        """Return an answer to a question about a patch."""
        ...