from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"{name} must be a boolean")
    return normalized in {"true", "1", "yes"}


def _csv(name: str, default: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in os.getenv(name, default).split(",") if part.strip())


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development").strip().lower()
    storage_dir: Path = Path(os.getenv("STORAGE_DIR", "storage"))
    detector_model: Path = Path(os.getenv("DETECTOR_MODEL", "models/key_detector.pt"))
    embedding_model: Path = Path(
        os.getenv("EMBEDDING_MODEL", "models/key_embedding_model_traced.pt")
    )
    match_threshold: float = float(os.getenv("MATCH_THRESHOLD", ".65"))
    consistency_threshold: float = float(os.getenv("CONSISTENCY_THRESHOLD", "0"))
    bitting_shortlist: int = int(os.getenv("BITTING_SHORTLIST", "10"))
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    max_image_pixels: int = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
    sqlite_timeout_seconds: float = float(os.getenv("SQLITE_TIMEOUT_SECONDS", "30"))
    allow_segmentation_fallback: bool = _bool("ALLOW_SEGMENTATION_FALLBACK", False)
    api_key: str | None = os.getenv("API_KEY") or None
    allowed_origins: tuple[str, ...] = _csv("ALLOWED_ORIGINS", "")
    allowed_hosts: tuple[str, ...] = _csv("ALLOWED_HOSTS", "localhost,127.0.0.1")
    enable_docs: bool = _bool("ENABLE_DOCS", True)

    def validate(self) -> None:
        if self.app_env not in {"development", "test", "production"}:
            raise ValueError("APP_ENV must be development, test, or production")
        if not 0 <= self.match_threshold <= 1:
            raise ValueError("MATCH_THRESHOLD must be between 0 and 1")
        if not 0 <= self.consistency_threshold <= 1:
            raise ValueError("CONSISTENCY_THRESHOLD must be between 0 and 1")
        if self.bitting_shortlist < 1:
            raise ValueError("BITTING_SHORTLIST must be positive")
        if self.max_upload_bytes < 1024:
            raise ValueError("MAX_UPLOAD_BYTES must be at least 1024")
        if self.max_image_pixels < 1:
            raise ValueError("MAX_IMAGE_PIXELS must be positive")
        if self.sqlite_timeout_seconds <= 0:
            raise ValueError("SQLITE_TIMEOUT_SECONDS must be positive")
        if not self.detector_model.is_file():
            raise ValueError(f"Detector model not found: {self.detector_model}")
        if not self.embedding_model.is_file():
            raise ValueError(f"Embedding model not found: {self.embedding_model}")
        if self.app_env == "production":
            if self.allow_segmentation_fallback:
                raise ValueError("ALLOW_SEGMENTATION_FALLBACK must be false in production")
            if not self.api_key or len(self.api_key) < 24:
                raise ValueError("Production API_KEY must contain at least 24 characters")

    @property
    def images_dir(self) -> Path:
        return self.storage_dir / "images"

    @property
    def artifacts_dir(self) -> Path:
        return self.storage_dir / "artifacts"

    @property
    def database_path(self) -> Path:
        return self.storage_dir / "features.sqlite3"


settings = Settings()
