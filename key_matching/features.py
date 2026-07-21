from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d


def extract_bitting_profile(blade_mask: np.ndarray, samples: int = 256) -> np.ndarray:
    """Return normalized upper/lower cutting-edge depths along the blade."""
    xs = np.flatnonzero(np.any(blade_mask > 0, axis=0))
    if xs.size < 8:
        raise ValueError("Blade mask is empty or too short")
    upper, lower = [], []
    for x in xs:
        ys = np.flatnonzero(blade_mask[:, x] > 0)
        upper.append(float(ys.min()) if ys.size else np.nan)
        lower.append(float(ys.max()) if ys.size else np.nan)
    upper = _interpolate(np.asarray(upper))
    lower = _interpolate(np.asarray(lower))
    center = np.median((upper + lower) * .5)
    scale = max(float(np.percentile(lower - upper, 90)), 1.0)
    positions = np.linspace(0, len(upper) - 1, samples)
    source = np.arange(len(upper))
    top_depth = np.interp(positions, source, (upper - center) / scale)
    bottom_depth = np.interp(positions, source, (center - lower) / scale)
    return np.stack((gaussian_filter1d(top_depth, 1),
                     gaussian_filter1d(bottom_depth, 1))).astype(np.float32)


def extract_blade_contour(blade_mask: np.ndarray, points: int = 256) -> np.ndarray:
    contours, _ = cv2.findContours(blade_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("Blade contour not found")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    delta = np.diff(np.vstack((contour, contour[:1])), axis=0)
    cumulative = np.r_[0, np.cumsum(np.linalg.norm(delta, axis=1))]
    targets = np.linspace(0, cumulative[-1], points, endpoint=False)
    closed = np.vstack((contour, contour[:1]))
    sampled = np.column_stack([np.interp(targets, cumulative, closed[:, i]) for i in range(2)])
    sampled -= sampled.mean(axis=0)
    sampled /= max(np.ptp(sampled[:, 0]), np.ptp(sampled[:, 1]), 1.0)
    return sampled.astype(np.float32)


def _interpolate(values: np.ndarray) -> np.ndarray:
    good = np.isfinite(values)
    if not np.any(good):
        return np.zeros_like(values)
    return np.interp(np.arange(values.size), np.flatnonzero(good), values[good])
