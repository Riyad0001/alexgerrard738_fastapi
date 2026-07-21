from typing import Any, Literal

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    key_id: str
    side: Literal["front", "back"]
    score: float = Field(ge=0, le=1)
    bitting_score: float = Field(ge=0, le=1)
    blade_score: float = Field(ge=0, le=1)
    embedding_score: float = Field(ge=0, le=1)
    registration: dict[str, Any]


class MatchResponse(BaseModel):
    query_id: str
    matches: list[Candidate]
    confident_match: bool
    threshold: float = Field(ge=0, le=1)


class RegisterResponse(BaseModel):
    key_id: str
    status: Literal["registered"]
    sides: list[Literal["front", "back"]]
    registration: dict[str, Any]
    consistency_score: float | None = Field(default=None, ge=0, le=1)


class KeyRecord(BaseModel):
    key_id: str
    metadata: dict[str, Any]
    sides: list[Literal["front", "back"]]
    created_at: str
    updated_at: str


class DeleteResponse(BaseModel):
    key_id: str
    status: Literal["deleted"]


# ── Django-compatible response shapes ─────────────────────────────────────────

class KeyInfo(BaseModel):
    """Mirrors Django KeySerializer output."""
    id: str
    key_type: str | None = None
    manufacturer: str | None = None
    lock_brand: str | None = None
    key_code: str | None = None
    key_bitting: str | None = None
    no_pins: int | None = None
    brand: str | None = None
    code: str | None = None
    x: str | None = None
    cross_refs: list[Any] = Field(default_factory=list)
    front_image: str | None = None
    back_image: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class MatchCandidateDjango(BaseModel):
    key_info: KeyInfo
    matched_side: str
    similarity_score: float
    confidence_percentage: float


class MatchResponseDjango(BaseModel):
    message: str
    best_match: MatchCandidateDjango | None = None
    all_matches: list[MatchCandidateDjango] = Field(default_factory=list)


class TrainedKeyItem(BaseModel):
    id: str
    title: str
    manufacturer: str | None = None
    key_code: str | None = None
    time_ago: str


class TrainedKeyGroup(BaseModel):
    date: str
    items: list[TrainedKeyItem]


class ReferencesResponse(BaseModel):
    key_id: str
    references: list[Any] = Field(default_factory=list)
