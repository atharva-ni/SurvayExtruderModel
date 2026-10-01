import numpy as np
import pandas as pd

from classifier import prepare_frame, categorize

ABSTRACT = "We study resource allocation in wireless networks and evaluate the approach with simulations " * 2
SURVEY_ABSTRACT = "In this survey, we review existing work on edge computing and outline open challenges " * 2


def frame(rows):
    return prepare_frame(pd.DataFrame(rows, columns=["title", "abstract", "venue", "type"]))


def test_prepare_frame_maps_columns_and_counts_references():
    df = pd.DataFrame({"title": ["T"], "abstract": [None], "venue": [None], "references": ["a; b; ; c"]})
    out = prepare_frame(df)
    assert out.loc[0, "Title"] == "T"
    assert out.loc[0, "Abstract"] == ""
    assert out.loc[0, "Venue"] == ""
    assert out.loc[0, "ReferenceCount"] == 3


def test_survey_and_research_pass_through():
    f = frame([["Edge Computing: A Comprehensive Survey", SURVEY_ABSTRACT, "IEEE Commun. Surv. Tutorials", "JournalArticle"],
               ["Optimal Power Allocation for NOMA", ABSTRACT, "IEEE Transactions on Communications", "JournalArticle"]])
    assert list(categorize(f, np.array([1, 0]))) == ["survey", "research"]


def test_non_paper_by_type_or_title_wins():
    f = frame([["Game Theory in Wireless Networks", ABSTRACT, "", "book"],
               ["Guest Editorial: Special Issue on 6G", ABSTRACT, "IEEE JSAC", "JournalArticle"]])
    assert list(categorize(f, np.array([1, 1]))) == ["non-paper", "non-paper"]


def test_title_only_rule():
    f = frame([
        # only a title, no survey keyword: not a survey even if the classifier says so (typically a book)
        ["Game Theory in Wireless and Communication Networks", "", "", ""],
        # only a title, with a survey keyword: survey
        ["A Survey on Software-Defined Networking", "", "", ""],
        # has a venue, so the classifier decides
        ["Mobile Cloud Computing Architectures", "", "Wireless Communications and Mobile Computing", ""],
    ])
    assert list(categorize(f, np.array([1, 0, 1]))) == ["research", "survey", "survey"]
