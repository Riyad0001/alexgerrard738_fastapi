from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import directed_hausdorff

from .types import KeyFeatures


@dataclass(frozen=True)
class MatchWeights:
    bitting: float = .60
    blade: float = .20
    embedding: float = .20

    def __post_init__(self) -> None:
        if min(self.bitting, self.blade, self.embedding) < 0:
            raise ValueError("Weights cannot be negative")
        if not np.isclose(self.bitting + self.blade + self.embedding, 1.0):
            raise ValueError("Weights must sum to one")


ADVERSE_VIEW_WEIGHTS = MatchWeights(bitting=.30, blade=.10, embedding=.60)


# def select_match_weights(bitting: float, embedding: float,
#                          default: MatchWeights) -> MatchWeights:
#     """Use appearance support only when geometry still provides agreement."""
#     if bitting >= .45 and embedding >= .75:
#         return ADVERSE_VIEW_WEIGHTS
#     return default
def select_match_weights(bitting: float, embedding: float,
                         default: MatchWeights) -> MatchWeights:
    """Always use the default weights to ensure consistent scoring."""
    return default

def bitting_similarity(query: np.ndarray, reference: np.ndarray) -> tuple[float, bool]:
    if query.shape != reference.shape or query.size == 0:
        return 0.0, False
    candidates = [(reference, False), (reference[::-1, :], True)]
    best = (0.0, False)
    for candidate, mirrored in candidates:
        # Small horizontal shifts tolerate imperfect shoulder localization.
        error = min(np.mean(np.abs(query[:, max(s, 0):query.shape[1] + min(s, 0)] -
                                   candidate[:, max(-s, 0):candidate.shape[1] - max(s, 0)]))
                    for s in range(-4, 5))
        score = float(np.exp(-error / .12))
        if score > best[0]:
            best = score, mirrored
    return best


def contour_similarity(query: np.ndarray, reference: np.ndarray) -> float:
    if query.ndim != 2 or reference.ndim != 2 or not query.size or not reference.size:
        return 0.0
    # Front/back views reflect the cutting edges vertically after the bow is
    # normalized to the left, so score both valid orientations.
    candidates = (reference, reference * np.array([1.0, -1.0], dtype=np.float32))
    distance = min(max(directed_hausdorff(query, candidate)[0],
                       directed_hausdorff(candidate, query)[0])
                   for candidate in candidates)
    return float(np.exp(-distance / .08))


def embedding_similarity(query: np.ndarray, reference: np.ndarray) -> float:
    if query.shape != reference.shape or query.size == 0:
        return 0.5  # neutral when the optional verification model is unavailable
    cosine = float(np.dot(query, reference) /
                   max(np.linalg.norm(query) * np.linalg.norm(reference), 1e-12))
    return float(np.clip((cosine + 1) * .5, 0, 1))


def score_pair(query: KeyFeatures, reference: KeyFeatures,
               weights: MatchWeights = MatchWeights()) -> tuple[float, dict[str, float | bool]]:
    bitting, mirrored = bitting_similarity(query.bitting_profile, reference.bitting_profile)
    contour = contour_similarity(query.blade_contour, reference.blade_contour)
    embedding = embedding_similarity(query.embedding, reference.embedding)
    applied = select_match_weights(bitting, embedding, weights)
    total = (applied.bitting * bitting + applied.blade * contour +
             applied.embedding * embedding)
    return float(total), {"bitting": bitting, "blade_contour": contour,
                          "embedding": embedding, "mirrored": mirrored,
                          "adverse_view": applied is ADVERSE_VIEW_WEIGHTS}


def rank_candidates(query: KeyFeatures, references: list[tuple[str, KeyFeatures]],
                    limit: int = 5, shortlist: int = 10,
                    weights: MatchWeights = MatchWeights()) -> list[dict]:
    """Shortlist by bitting, then apply blade and embedding verification."""
    coarse = []
    for reference_id, features in references:
        score, _ = bitting_similarity(query.bitting_profile, features.bitting_profile)
        coarse.append((score, reference_id, features))
    coarse.sort(key=lambda row: row[0], reverse=True)
    ranked = []
    for _, reference_id, features in coarse[:shortlist]:
        score, breakdown = score_pair(query, features, weights)
        ranked.append({"reference_id": reference_id, "score": score, "breakdown": breakdown})
    return sorted(ranked, key=lambda row: row["score"], reverse=True)[:limit]
