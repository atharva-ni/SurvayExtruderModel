import numpy as np
import pandas as pd

from classifier import prepare_frame
from learned_hybrid import (LearnedHybrid, best_threshold, build_features, hybrid_features, weighted_f1,
                      weighted_prec_rec)

ABSTRACT = "In this article, we propose a resource allocation scheme and evaluate it with simulations " * 2


def test_weighted_precision_recall_uses_weights():
    y = np.array([1, 1, 0, 0])
    p = np.array([1, 0, 1, 0])
    w = np.array([1.0, 1.0, 3.0, 1.0])
    prec, rec = weighted_prec_rec(y, p, w)
    assert prec == 0.25 and rec == 0.5
    assert abs(weighted_f1(y, p, w) - 2 * 0.25 * 0.5 / 0.75) < 1e-12


def test_best_threshold_separates_scores():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    t = best_threshold(y, scores, np.ones(4))
    assert 0.2 < t <= 0.8


def test_hybrid_features_flags_and_embedding():
    f = prepare_frame(pd.DataFrame(
        [["An Overview", ABSTRACT, "IEEE Communications Magazine", "JournalArticle; Review"],
         ["A Scheme", ABSTRACT, "arXiv.org", "JournalArticle"]],
        columns=["title", "abstract", "venue", "type"]))
    X = hybrid_features(f, np.array([0.9, 0.1]), np.zeros((2, 3)))
    flags = X[:, -7:-3]  # magazine, arxiv, type_review, this_article
    assert flags[0].tolist() == [1, 0, 1, 1]
    assert flags[1].tolist() == [0, 1, 0, 1]
    assert X.shape[1] == 7 + 4 + 3


def test_fit_selected_learns_simple_rule():
    rng = np.random.default_rng(0)
    X_fixed = rng.normal(size=(200, 3))
    y_fixed = (X_fixed[:, 0] > 0).astype(int)
    X_hand = rng.normal(size=(60, 3))
    y_hand = (X_hand[:, 0] > 0).astype(int)
    comb = LearnedHybrid.fit_selected(X_fixed, y_fixed, np.ones(200), X_hand, y_hand, np.ones(60),
                                       np.array(["s"] * 60))
    assert (comb.predict(np.array([[2.0, 0, 0], [-2.0, 0, 0]])) == [1, 0]).all()


def test_build_features_marks_missing_refs_and_abstract():
    df = pd.DataFrame({"Title": ["A Survey on X", "Scheme for Y"],
                       "Abstract": ["", "We propose a scheme and simulation results show gains " * 2],
                       "ReferenceCount": [150, 0]})
    X = build_features(df, np.array([0.9, 0.1]))
    assert list(X["abstract_missing"]) == [1, 0]
    assert list(X["refs_missing"]) == [0, 1]
    assert list(X["title_terms"]) == [1, 0]
    assert X.loc[1, "abstract_research_cues"] >= 1
