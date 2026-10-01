import numpy as np
import pandas as pd
import pytest

from classifier import calculate_indices
from train import best_f1_threshold


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


def test_corrected_shares_inverts_error_rates():
    from evaluate import corrected_shares
    # A perfect classifier needs no correction
    pi, cites = corrected_shares(0.10, 0.30, tpr=1.0, fpr=0.0)
    assert abs(pi - 0.10) < 1e-12 and abs(cites - 0.30) < 1e-12
    # 20% true surveys, recall 0.8, FPR 0.01: flagged share 0.8 * 0.2 + 0.01 * 0.8 = 0.168
    pi, _ = corrected_shares(0.168, 0.5, tpr=0.8, fpr=0.01)
    assert abs(pi - 0.20) < 1e-9
    # Flagging no more than the false-positive rate means no surveys
    assert corrected_shares(0.005, 0.1, tpr=0.8, fpr=0.01)[0] == 0


def test_baseline_rules():
    from classifier import prepare_frame
    from evaluate import original_tool, smyth_cunningham_rule
    f = prepare_frame(pd.DataFrame([
        ["Coverage Analysis of mmWave Networks", "We derive coverage.", "IEEE TWC", "JournalArticle"],
        ["Edge AI: A Comprehensive Survey", "We cover edge AI.", "IEEE Commun. Surv. Tutorials", "JournalArticle; Review"],
        ["Edge AI: Challenges", "This article gives an overview of edge AI.", "IEEE Network", "JournalArticle; Review"],
        ["Edge AI: Challenges", "This article discusses edge AI.", "IEEE Network", "JournalArticle; Review"],
        ["A Survey of Edge AI", "A survey.", "IEEE Access", "JournalArticle"],
    ], columns=["title", "abstract", "venue", "type"]))
    # Substring match of the original phrases, so 'Analysis of' flags a research paper
    assert list(original_tool(f)) == [1, 1, 1, 1, 1]
    # Needs the Review type AND a strict phrase or a review venue
    assert list(smyth_cunningham_rule(f)) == [0, 1, 1, 0, 0]
