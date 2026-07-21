from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .types import SegmentationResult

LOGGER = logging.getLogger(__name__)


class SegmentationError(ValueError):
    pass


class KeySegmenter:
    """YOLO segmenter with adaptive confidence and an optional CV fallback."""

    def __init__(self, model_path: str | None, confidence: float = 0.4,
                 retry_confidences: tuple[float, ...] = (0.25, 0.15),
                 allow_fallback: bool = True) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.retry_confidences = retry_confidences
        self.allow_fallback = allow_fallback
        self._model = None

    def _load_model(self):
        if self._model is None and self.model_path and Path(self.model_path).exists():
            try:
                from ultralytics import YOLO
                self._model = YOLO(self.model_path)
            except (ImportError, RuntimeError) as exc:
                if not self.allow_fallback:
                    raise SegmentationError("YOLO segmentation is unavailable") from exc
                LOGGER.warning("YOLO unavailable; using threshold fallback: %s", exc)
        return self._model

    def segment(self, image: np.ndarray) -> SegmentationResult:
        model = self._load_model()
        if model is None:
            if not self.allow_fallback:
                raise SegmentationError("Segmentation model not found")
            return self._classical(image)

        candidates: list[tuple[np.ndarray, float]] = []
        attempted = (self.confidence, *self.retry_confidences)
        for threshold in attempted:
            result = model.predict(image, conf=threshold, verbose=False)[0]
            if result.masks is None:
                continue
            candidates = [
                (mask.cpu().numpy(), float(box.conf.item()))
                for mask, box in zip(result.masks.data, result.boxes)
                if int(box.cls.item()) == 0
            ]
            if candidates:
                break

        if not candidates:
            tried = ", ".join(f"{value:.2f}" for value in attempted)
            if self.allow_fallback:
                LOGGER.warning("YOLO found no key at %s; trying CV fallback", tried)
                return self._classical(image)
            raise SegmentationError(
                f"YOLO found no key mask (confidence thresholds tried: {tried})")
        if len(candidates) > 1:
            raise SegmentationError(
                f"Expected one key, but YOLO found {len(candidates)} keys")

        mask, confidence = candidates[0]
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        return self._result(image, mask > 0.5, confidence)

    @staticmethod
    def _result(image: np.ndarray, mask: np.ndarray, confidence: float) -> SegmentationResult:
        binary = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise SegmentationError("Detected key mask was empty")
        x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
        crop = cv2.bitwise_and(image, image, mask=binary)[y:y + height, x:x + width]
        return SegmentationResult(binary, crop, confidence)

    def _classical(self, image: np.ndarray) -> SegmentationResult:
        gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        image_area = gray.size
        candidates = []
        for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            _, mask = cv2.threshold(gray, 0, 255, mode | cv2.THRESH_OTSU)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                _, (width, height), _ = cv2.minAreaRect(contour)
                if (.005 * image_area < area < .75 * image_area and
                        min(width, height) > 0 and
                        max(width, height) / min(width, height) >= 1.5):
                    candidates.append((area, contour))
        if not candidates:
            raise SegmentationError("No key detected by YOLO or CV fallback")
        contour = max(candidates, key=lambda item: item[0])[1]
        final = np.zeros(gray.shape, np.uint8)
        cv2.drawContours(final, [contour], -1, 255, cv2.FILLED)
        return self._result(image, final > 0, 0.5)
