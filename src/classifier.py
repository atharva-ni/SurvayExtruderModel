"""
Survey/Non-Research Paper Classifier
====================================
Classifies an author's publications, excludes surveys, and recalculates
h-index, i10-index and citations. Non-papers (editorials, errata, ...) stay in
both the original and the filtered metrics.

Pipeline: DistilBERT scores title + abstract, the learned hybrid (src/learned_hybrid.py)
combines that score with DistilBERT's [CLS] representation and metadata
features, and decision rules assign one category per paper:
  * non-paper - books, editorials, errata, ... (by publication type or title)
  * survey    - detected survey
  * research  - everything else
"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

from text_utils import NON_PAPER_TITLE, NON_PAPER_TYPES, paper_text, clean_text, keyword_is_survey
from inference import load_model, load_config, predict_survey_proba
from learned_hybrid import LearnedHybrid, cls_embeddings, hybrid_features


# ============================================================================
# INPUT NORMALIZATION
# ============================================================================

def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map Semantic Scholar / dataset column names to Title, Abstract, ReferenceCount."""
    out = pd.DataFrame(index=df.index)
    out["Title"] = df.get("Title", df.get("title", pd.Series("", index=df.index))).fillna("").astype(str)
    out["Abstract"] = df.get("Abstract", df.get("abstract", pd.Series("", index=df.index))).fillna("").astype(str)

    if "ReferenceCount" in df.columns:
        out["ReferenceCount"] = pd.to_numeric(df["ReferenceCount"], errors="coerce").fillna(0)
    elif "references" in df.columns:
        out["ReferenceCount"] = df["references"].fillna("").map(
            lambda s: len([r for r in str(s).split(";") if r.strip()]))
    else:
        out["ReferenceCount"] = 0

    out["Venue"] = df.get("Venue", df.get("venue", pd.Series("", index=df.index))).fillna("").astype(str)
    out["Type"] = df.get("Type", df.get("type", pd.Series("", index=df.index))).fillna("").astype(str)
    return out


# ============================================================================
# CLASSIFICATION
# ============================================================================

def classify_frame(
    frame: pd.DataFrame,
    model_path: str = "./distilbert_survey_model",
    batch_size: int = 32,
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Learned hybrid. Returns (is_survey, score) arrays; is_survey: 1 = survey, 0 = not a survey."""
    hybrid = LearnedHybrid.load(model_path)
    if hybrid is None:
        raise FileNotFoundError(f"No learned hybrid in '{model_path}' — fit it with: python main.py train --hybrid-only")

    config = load_config(model_path)
    tokenizer, model = load_model(model_path)
    texts = [paper_text(t, a) for t, a in zip(frame["Title"], frame["Abstract"])]
    survey_proba = predict_survey_proba(texts, tokenizer, model, batch_size=batch_size,
                                        max_length=config.get("max_length", 384), show_progress=show_progress)
    X = hybrid_features(frame, survey_proba, cls_embeddings(frame, model_path, batch_size))
    scores = hybrid.predict_proba(X)
    return (scores >= hybrid.threshold).astype(int), scores


def is_non_paper(frame: pd.DataFrame) -> np.ndarray:
    """Books, editorials, errata, ...: by publication type when known, otherwise by title."""
    by_title = frame["Title"].map(lambda t: bool(NON_PAPER_TITLE.search(clean_text(t))))
    by_type = frame["Type"].map(lambda t: bool(NON_PAPER_TYPES.search(t)))
    return (by_title | by_type).values


def has_no_metadata(frame: pd.DataFrame) -> np.ndarray:
    """Only a title is known (no abstract, venue or publication type): typically books and chapters."""
    no_abstract = frame["Abstract"].map(lambda a: len(clean_text(a).split()) < 10)
    return (no_abstract & (frame["Venue"].str.strip() == "") & (frame["Type"].str.strip() == "")).values


def categorize(frame: pd.DataFrame, is_survey: np.ndarray) -> np.ndarray:
    """Return one of: non-paper, survey, research."""
    # With only a title, book titles look like overviews; require an explicit survey keyword instead
    title_only = has_no_metadata(frame)
    keyword = np.array([keyword_is_survey(t) for t in frame["Title"]], dtype=int)
    is_survey = np.where(title_only, keyword, is_survey)
    return np.where(is_non_paper(frame), "non-paper", np.where(is_survey == 1, "survey", "research"))


# ============================================================================
# INDEX CALCULATION
# ============================================================================

def calculate_indices(df: pd.DataFrame) -> Tuple[int, int]:
    """Calculate h-index and i10-index from citation counts."""
    citations = df["citationCount"].fillna(0).astype(int).sort_values(ascending=False).values
    h_index = int(sum(c >= (i + 1) for i, c in enumerate(citations)))
    i10_index = int(sum(c >= 10 for c in citations))
    return h_index, i10_index


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_classification_pipeline(
    input_csv: str,
    output_csv: str,
    survey_csv: str,
    model_path: str = "./distilbert_survey_model",
    batch_size: int = 32,
) -> dict:
    """Classify papers, save research / excluded papers, and print metric changes."""
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required_cols = {"title", "abstract", "citationCount"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"CSV must contain: {required_cols}")
    df["citationCount"] = df["citationCount"].fillna(0).astype(int)
    print(f"✅ Loaded {len(df)} papers from '{input_csv}'")

    frame = prepare_frame(df)
    is_survey, score = classify_frame(frame, model_path=model_path, batch_size=batch_size, show_progress=True)
    df["Category"] = categorize(frame, is_survey)
    df["SurveyScore"] = np.round(score, 4)
    df["Keep"] = (df["Category"] != "survey").astype(int)  # 1 = kept in the metrics, 0 = excluded as a survey

    # Only surveys are removed from the metrics; non-papers stay in both profiles
    excluded_df = df[df["Keep"] == 0]
    filtered_df = df[df["Keep"] == 1]
    research_df = filtered_df[filtered_df["Category"] != "non-paper"]
    other_df = df[~df.index.isin(research_df.index)]

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(survey_csv)), exist_ok=True)
    research_df.to_csv(output_csv, index=False)
    other_df.to_csv(survey_csv, index=False)

    # Statistics
    total_papers, total_citations = len(df), int(df["citationCount"].sum())
    excluded_citations = int(excluded_df["citationCount"].sum())
    total_h, total_i10 = calculate_indices(df)
    filtered_h, filtered_i10 = calculate_indices(filtered_df)
    n_non_papers = int((df["Category"] == "non-paper").sum())

    print(f"\n📊 Surveys excluded: {len(excluded_df)} ({100 * len(excluded_df) / max(1, total_papers):.2f}%)")
    if n_non_papers:
        print(f"ℹ️  {n_non_papers} books/editorials/other non-papers are kept in both metrics "
              f"but left out of the research-only file")
    print(f"📉 Citations excluded: {excluded_citations} ({100 * excluded_citations / max(1, total_citations):.2f}%)")

    comparison = [
        ["Total Papers", total_papers, len(filtered_df)],
        ["Total Citations", total_citations, total_citations - excluded_citations],
        ["H-Index", total_h, filtered_h],
        ["i10-Index", total_i10, filtered_i10],
    ]
    print("\n📋 Comparison Table:")
    print(tabulate(comparison, headers=["Metric", "All Papers", "Without Surveys"], tablefmt="grid"))
    print(f"\n✅ Research papers saved to: '{output_csv}'")
    print(f"✅ Surveys and non-papers saved to: '{survey_csv}'")

    return {
        "papers": total_papers, "surveys": len(excluded_df), "non_papers": n_non_papers,
        "citations": total_citations, "excluded_citations": excluded_citations,
        "h_before": total_h, "h_after": filtered_h, "i10_before": total_i10, "i10_after": filtered_i10,
    }
