import cv2
import numpy as np

from key_matching.features import extract_bitting_profile, extract_blade_contour
from key_matching.matcher import (MatchWeights, contour_similarity,
                                  rank_candidates, score_pair,
                                  select_match_weights)
from key_matching.types import KeyFeatures


def _features(notch: int = 25) -> KeyFeatures:
    mask = np.zeros((80, 320), np.uint8)
    cv2.rectangle(mask, (0, 20), (319, 60), 255, -1)
    cv2.rectangle(mask, (notch, 20), (notch + 20, 35), 0, -1)
    return KeyFeatures(extract_bitting_profile(mask, 128),
                       extract_blade_contour(mask, 128), np.ones(512) / np.sqrt(512))


def test_identical_geometry_scores_higher_than_different_bitting():
    query = _features(30)
    same, breakdown = score_pair(query, _features(30))
    different, _ = score_pair(query, _features(180))
    assert same > different
    assert breakdown["bitting"] == 1.0


def test_ranking_shortlists_by_bitting():
    query = _features(30)
    ranked = rank_candidates(query, [("wrong", _features(180)), ("right", _features(30))])
    assert ranked[0]["reference_id"] == "right"


def test_weights_must_sum_to_one():
    try:
        MatchWeights(.5, .2, .2)
    except ValueError:
        return
    raise AssertionError("invalid weights accepted")


def test_contour_similarity_accepts_front_back_reflection():
    contour = _features(30).blade_contour
    reflected = contour * np.array([1.0, -1.0], dtype=np.float32)

    assert contour_similarity(contour, reflected) == 1.0


def test_adverse_view_requires_geometry_and_embedding_agreement():
    default = MatchWeights()

    assert select_match_weights(.53, .81, default).embedding == .60
    assert select_match_weights(.30, .81, default) is default
    assert select_match_weights(.53, .65, default) is default
