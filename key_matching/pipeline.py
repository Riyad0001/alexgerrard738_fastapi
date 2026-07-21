from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import BinaryIO

from .detector import KeySegmenter, SegmentationError
from .embeddings import EmbeddingExtractor
from .features import extract_bitting_profile, extract_blade_contour
from .geometry import GeometryError, normalize_geometry
from .io import load_image
from .types import KeyFeatures, NormalizedKey


class PipelineError(ValueError):
    pass


class KeyPipeline:
    def __init__(self, detector_path: str | None = "models/key_detector.pt",
                 embedding_path: str | None = "models/key_embedding_model_traced.pt",
                 blade_length: int = 1024, allow_segmentation_fallback: bool = False,
                 max_image_pixels: int = 40_000_000) -> None:
        self.segmenter = KeySegmenter(detector_path, allow_fallback=allow_segmentation_fallback)
        self.embedder = EmbeddingExtractor(embedding_path)
        self.blade_length = blade_length
        self.max_image_pixels = max_image_pixels
        self._lock = RLock()

    def load_models(self) -> None:
        if self.segmenter._load_model() is None:
            raise PipelineError("YOLO segmentation model could not be loaded")
        if self.embedder._load() is None:
            raise PipelineError("Embedding model could not be loaded")

    def process(self, source: str | Path | BinaryIO | bytes) -> tuple[KeyFeatures, NormalizedKey]:
        try:
            image = load_image(source, self.max_image_pixels)
            with self._lock:
                segmented = self.segmenter.segment(image)
                normalized = normalize_geometry(image, segmented.mask, self.blade_length)
                features = KeyFeatures(
                    bitting_profile=extract_bitting_profile(normalized.blade_mask),
                    blade_contour=extract_blade_contour(normalized.blade_mask),
                    embedding=self.embedder.extract(normalized.image, normalized.mask),
                    metadata={"segmentation_confidence": segmented.confidence,
                              "perspective_confidence": normalized.perspective_confidence,
                              "blade_start": normalized.blade_start},
                )
            return features, normalized
        except (OSError, ValueError, SegmentationError, GeometryError) as exc:
            raise PipelineError(str(exc)) from exc
