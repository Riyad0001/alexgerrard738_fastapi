import cv2
import numpy as np

from key_matching.geometry import find_blade_start, normalize_geometry


def test_geometry_ignores_disconnected_mask_noise():
    image = np.zeros((240, 640, 3), dtype=np.uint8)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (100, 120), 70, 255, -1)
    cv2.rectangle(mask, (100, 100), (590, 140), 255, -1)
    mask[5, 5] = 255
    image[mask > 0] = 180

    normalized = normalize_geometry(image, mask, blade_length=256)

    assert normalized.blade_mask.any()
    assert normalized.perspective_confidence >= 0.45


def test_blade_start_ignores_narrow_leading_edge_of_round_bow():
    mask = np.zeros((200, 600), dtype=np.uint8)
    cv2.circle(mask, (130, 100), 100, 255, -1)
    cv2.rectangle(mask, (130, 80), (599, 120), 255, -1)

    assert find_blade_start(mask) == mask.shape[1] // 3
