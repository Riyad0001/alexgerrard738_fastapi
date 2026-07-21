from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pillow_heif = None

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def load_image(source: str | Path | BinaryIO | bytes,
               max_pixels: int = 40_000_000) -> np.ndarray:
    """Load a supported image, apply EXIF orientation, and return BGR uint8."""
    if isinstance(source, bytes):
        stream = io.BytesIO(source)
    elif hasattr(source, "read"):
        stream = source
    else:
        path = Path(source)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix}")
        stream = path
    with Image.open(stream) as image:
        if image.width * image.height > max_pixels:
            raise ValueError(
                f"Image dimensions exceed the {max_pixels}-pixel limit"
            )
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
