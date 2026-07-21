from __future__ import annotations

import cv2
import numpy as np

from .types import NormalizedKey


class GeometryError(ValueError):
    pass


def normalize_geometry(image: np.ndarray, mask: np.ndarray,
                       blade_length: int = 1024,
                       min_perspective_confidence: float = 0.45) -> NormalizedKey:
    """Rectify to a horizontal bow-left/tip-right canonical frame."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise GeometryError("Key mask is empty")
    # One physical key is enforced by YOLO instance count. A single instance
    # mask can still contain detached edge/glare pixels, so use its dominant
    # component rather than treating mask noise as another physical key.
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 0.005 * mask.size:
        raise GeometryError("Detected key mask is too small")
    clean_mask = np.zeros_like(mask)
    cv2.drawContours(clean_mask, [contour], -1, 255, cv2.FILLED)

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect).astype(np.float32)
    width, height = rect[1]
    long_side, short_side = max(width, height), min(width, height)
    if short_side < 2 or long_side / short_side < 1.5:
        raise GeometryError("Object is not key-shaped")
    box = _ordered_box(box, horizontal=width >= height)
    target_height = max(64, int(round(short_side)))
    destination = np.array(
        [[0, 0], [long_side - 1, 0],
         [long_side - 1, target_height - 1], [0, target_height - 1]],
        np.float32,
    )
    transform = cv2.getPerspectiveTransform(box, destination)
    output_size = (max(int(round(long_side)), 2), target_height)
    corrected = cv2.warpPerspective(image, transform, output_size)
    corrected_mask = cv2.warpPerspective(
        clean_mask, transform, output_size, flags=cv2.INTER_NEAREST)

    fill = cv2.contourArea(contour) / max(long_side * short_side, 1)
    confidence = float(np.clip(fill / 0.55, 0, 1))
    if confidence < min_perspective_confidence:
        raise GeometryError(f"Perspective confidence too low: {confidence:.2f}")

    midpoint = corrected_mask.shape[1] // 2
    if (np.count_nonzero(corrected_mask[:, midpoint:]) >
            np.count_nonzero(corrected_mask[:, :midpoint])):
        corrected = cv2.rotate(corrected, cv2.ROTATE_180)
        corrected_mask = cv2.rotate(corrected_mask, cv2.ROTATE_180)

    blade_start = find_blade_start(corrected_mask)
    current_length = corrected_mask.shape[1] - blade_start
    scale = blade_length / max(current_length, 1)
    size = (max(int(round(corrected.shape[1] * scale)), 2),
            max(int(round(corrected.shape[0] * scale)), 2))
    corrected = cv2.resize(corrected, size, interpolation=cv2.INTER_LINEAR)
    corrected_mask = cv2.resize(corrected_mask, size, interpolation=cv2.INTER_NEAREST)
    blade_start = int(round(blade_start * scale))
    blade_mask = np.zeros_like(corrected_mask)
    blade_mask[:, blade_start:] = corrected_mask[:, blade_start:]
    return NormalizedKey(corrected, corrected_mask, blade_mask,
                         blade_start, confidence)


def _ordered_box(box: np.ndarray, horizontal: bool) -> np.ndarray:
    sums = box.sum(axis=1)
    differences = np.diff(box, axis=1).ravel()
    ordered = np.array(
        [box[np.argmin(sums)], box[np.argmin(differences)],
         box[np.argmax(sums)], box[np.argmax(differences)]],
        np.float32,
    )
    if not horizontal:
        ordered = ordered[[3, 0, 1, 2]]
    return ordered


def find_blade_start(mask: np.ndarray) -> int:
    length = mask.shape[1]
    # Rings touching the bow and wide teeth make transition-based shoulder
    # detection unstable across viewpoints. Rectification gives us a
    # canonical long axis; use its stable one-third boundary for the blade.
    return length // 3
