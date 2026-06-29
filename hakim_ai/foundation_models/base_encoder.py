"""
Abstract base class for all foundation model adapters.

Pattern: every real model (UNI 2, CONCH, PathChat, Virchow 2) implements
BaseEncoder / BaseVLM. Mock adapters return deterministic synthetic vectors
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

    @abc.abstractmethod
    def encode_patch(self, patch: Any) -> List[float]:
        """Return a 1-D feature vector for a single patch (PIL Image or np.ndarray)."""
        ...

    @abc.abstractmethod
    def encode_batch(self, patches: List[Any]) -> List[List[float]]:
        """Return a list of feature vectors for a batch of patches."""
        ...


class BaseVLM(abc.ABC):
    """Abstract vision-language model for patch description."""

    @abc.abstractmethod
    def describe_patch(self, patch: Any, prompt: str = "") -> str:
        """Return a natural-language description of a histopathology patch."""
        ...

    @abc.abstractmethod
    def answer_question(self, patch: Any, question: str) -> str:
        """Return an answer to a question about a patch."""
        ...