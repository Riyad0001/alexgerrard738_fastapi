from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    cropped_key: np.ndarray
    confidence: float


@dataclass(frozen=True)
class NormalizedKey:
    image: np.ndarray
    mask: np.ndarray
    blade_mask: np.ndarray
    blade_start: int
    perspective_confidence: float


@dataclass
class KeyFeatures:
    bitting_profile: np.ndarray
    blade_contour: np.ndarray
    embedding: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "bitting_profile": self.bitting_profile.astype(float).tolist(),
            "blade_contour": self.blade_contour.astype(float).tolist(),
            "embedding": self.embedding.astype(float).tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "KeyFeatures":
        return cls(
            bitting_profile=np.asarray(value["bitting_profile"], dtype=np.float32),
            blade_contour=np.asarray(value["blade_contour"], dtype=np.float32),
            embedding=np.asarray(value.get("embedding", []), dtype=np.float32),
            metadata=dict(value.get("metadata", {})),
        )
