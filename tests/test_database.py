import numpy as np
import pytest

from key_matching.database import DuplicateKeyError, FeatureStore
from key_matching.types import KeyFeatures


def _features() -> KeyFeatures:
    return KeyFeatures(
        bitting_profile=np.zeros((2, 4), dtype=np.float32),
        blade_contour=np.zeros((4, 2), dtype=np.float32),
        embedding=np.ones(4, dtype=np.float32) / 2,
    )


def test_registration_is_persisted_and_duplicate_is_rejected(tmp_path):
    store = FeatureStore(tmp_path / "features.sqlite3")
    metadata = {"key_type": "house", "num_pins": 5}
    sides = [
        (
            "front",
            _features(),
            {
                "front_image": "storage/images/front.jpg",
                "back_image": None,
                "normalized_image": "storage/artifacts/front.png",
                "mask_image": "storage/artifacts/front_mask.png",
            },
        )
    ]

    store.register("key-1", metadata, sides)
    record = store.get_registration("key-1")

    assert record is not None
    assert record["metadata"] == metadata
    assert record["sides"] == ["front"]
    assert store.list_registrations(10, 0)[0]["key_id"] == "key-1"
    with pytest.raises(DuplicateKeyError):
        store.register("key-1", metadata, sides)
    assert len(store.all()) == 1
    store.close()


def test_delete_removes_registration_and_returns_owned_paths(tmp_path):
    store = FeatureStore(tmp_path / "features.sqlite3")
    store.register(
        "key-1",
        {"key_type": "house"},
        [
            (
                "front",
                _features(),
                {
                    "front_image": "storage/images/front.jpg",
                    "back_image": None,
                    "normalized_image": "storage/artifacts/front.png",
                    "mask_image": "storage/artifacts/front_mask.png",
                },
            )
        ],
    )

    paths = store.delete("key-1")

    assert len(paths) == 3
    assert store.get_registration("key-1") is None
    assert store.all() == []
    assert store.delete("missing") == []
    store.close()
