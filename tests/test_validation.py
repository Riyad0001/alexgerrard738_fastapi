import io

import pytest
from fastapi import HTTPException
from PIL import Image

from key_matching.io import load_image
from key_matching.production_api import _normalize_bitting


def test_bitting_is_canonicalized_and_must_match_pin_count():
    assert _normalize_bitting("5, 2 4-7-6", 5) == "5-2-4-7-6"

    with pytest.raises(HTTPException) as error:
        _normalize_bitting("5-2-4", 5)
    assert error.value.status_code == 422


def test_image_pixel_limit_is_enforced_before_decode():
    stream = io.BytesIO()
    Image.new("RGB", (20, 20)).save(stream, format="PNG")
    stream.seek(0)

    with pytest.raises(ValueError, match="pixel limit"):
        load_image(stream, max_pixels=399)
