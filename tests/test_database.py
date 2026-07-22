import os
import uuid
import numpy as np
import pytest

from key_matching.database import DuplicateKeyError, FeatureStore
from key_matching.types import KeyFeatures

# Load database settings dynamically from environment variables
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


def _get_store() -> FeatureStore:
    if not DATABASE_URL or not (DATABASE_URL.startswith("postgresql://") or DATABASE_URL.startswith("postgres://")):
        pytest.skip("PostgreSQL test database not configured via DATABASE_URL environment variable")
    try:
        return FeatureStore(DATABASE_URL)
    except Exception as exc:
        pytest.skip(f"Failed to connect to PostgreSQL test database: {exc}")


def _features() -> KeyFeatures:
    return KeyFeatures(
        bitting_profile=np.zeros((2, 4), dtype=np.float32),
        blade_contour=np.zeros((4, 2), dtype=np.float32),
        embedding=np.ones(4, dtype=np.float32) / 2,
    )


def test_registration_is_persisted_and_duplicate_is_rejected():
    store = _get_store()
    key_id = f"test-key-{uuid.uuid4()}"
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

    try:
        store.register(key_id, metadata, sides)
        record = store.get_registration(key_id)

        assert record is not None
        assert record["metadata"] == metadata
        assert record["sides"] == ["front"]
        
        # Verify the key is listed
        registrations = store.list_registrations(10, 0)
        assert any(r["key_id"] == key_id for r in registrations)

        with pytest.raises(DuplicateKeyError):
            store.register(key_id, metadata, sides)
    finally:
        store.delete(key_id)
        store.close()


def test_delete_removes_registration_and_returns_owned_paths():
    store = _get_store()
    key_id = f"test-key-{uuid.uuid4()}"
    try:
        store.register(
            key_id,
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

        paths = store.delete(key_id)

        assert len(paths) == 3
        assert store.get_registration(key_id) is None
        assert store.delete("missing") == []
    finally:
        store.delete(key_id)
        store.close()
