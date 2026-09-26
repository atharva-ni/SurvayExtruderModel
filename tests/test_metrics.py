import numpy as np
import pandas as pd
import pytest

from classifier import calculate_indices
from hybrid import best_f1_threshold, or_rule, build_features


@pytest.mark.parametrize("citations, h, i10", [
    ([10, 8, 5, 4, 3], 4, 1),
    ([25, 8, 5, 3, 3], 3, 1),
    ([3, 3, 3], 3, 0),
    ([100], 1, 1),
    ([0, 0], 0, 0),
    ([], 0, 0),
])
def test_calculate_indices(citations, h, i10):
    assert calculate_indices(pd.DataFrame({"citationCount": citations})) == (h, i10)


def test_calculate_indices_ignores_order_and_missing_values():
    df = pd.DataFrame({"citationCount": [3, None, 10, 5, 4, 8]})
    assert calculate_indices(df) == (4, 1)


def test_best_f1_threshold_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    t = best_f1_threshold(y, scores)
    assert 0.3 < t <= 0.7


def test_or_rule_uses_model_or_title_keyword():
    proba = np.array([0.9, 0.1, 0.1])
    titles = ["Optimal Beamforming", "A Survey on RIS", "Optimal Beamforming"]
    assert list(or_rule(proba, titles, threshold=0.5)) == [1, 1, 0]


def test_build_features_marks_missing_refs_and_abstract():
    df = pd.DataFrame({"Title": ["A Survey on X", "Scheme for Y"],
                       "Abstract": ["", "We propose a scheme and simulation results show gains " * 2],
                       "ReferenceCount": [150, 0]})
    X = build_features(df, np.array([0.9, 0.1]))
    assert list(X["abstract_missing"]) == [1, 0]
    assert list(X["refs_missing"]) == [0, 1]
    assert list(X["title_terms"]) == [1, 0]
    assert X.loc[1, "abstract_research_cues"] >= 1
    # refs can be switched off (title + abstract only)
    assert list(build_features(df, np.array([0.9, 0.1]), use_refs=False)["refs_missing"]) == [1, 1]
