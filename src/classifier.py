"""
Survey/Non-Research Paper Classifier
====================================
Classifies an author's publications, excludes surveys, and recalculates
h-index, i10-index and citations. Non-papers (editorials, errata, ...) stay in
both the original and the filtered metrics.

Modes:
  * learned           - learned hybrid (default): logistic regression over the DistilBERT
                        score, its [CLS] representation and metadata features, fitted on the
                        validation split plus LLM- and hand-labeled author-profile papers
                        (src/combiner.py); falls back to validation-hybrid if not trained
  * validation-hybrid - earlier learned hybrid, fitted on the validation split only
  * or                - DistilBERT OR title keyword filter
  * model   - DistilBERT only
  * keyword - title keyword filter only

Categories:
  * non-paper         - books, editorials, errata, ... (by publication type or title)
  * survey            - detected survey
  * magazine-overview - only with --magazine-rule (default for modes other than learned):
                        flagged by the classifier, but a magazine article that does not
                        describe itself as a survey/tutorial/overview/review. Kept as
                        research unless --exclude-magazine-overviews is given
  * research          - everything else
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tabulate import tabulate

from text_utils import (NON_PAPER_TITLE, NON_PAPER_TYPES, MAGAZINE_VENUE, TITLE_SURVEY_TERMS,
                        EXPLICIT_SURVEY_CUES, paper_text, clean_text)
from inference import load_model, load_config, predict_survey_proba
from hybrid import LearnedHybrid, or_rule, keyword_is_survey
from combiner import SurveyCombiner, cls_embeddings, combiner_features

MODES = ("learned", "validation-hybrid", "or", "model", "keyword")
# The learned hybrid is fitted on labeled magazine articles, so it decides them itself
DEFAULT_MAGAZINE_RULE = False


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
    mode: str = "learned",
    threshold: Optional[float] = None,
    batch_size: int = 32,
    use_refs: bool = True,
    show_progress: bool = False,
    survey_proba: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (is_survey, score) arrays. is_survey: 1 = survey, 0 = not a survey.
    Pass survey_proba to reuse DistilBERT probabilities computed earlier.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")

    if mode == "keyword":
        is_survey = np.array([keyword_is_survey(t) for t in frame["Title"]], dtype=int)
        return is_survey, is_survey.astype(float)

    config = load_config(model_path)
    threshold = config["threshold"] if threshold is None else threshold

    if survey_proba is None:
        tokenizer, model = load_model(model_path)
        texts = [paper_text(t, a) for t, a in zip(frame["Title"], frame["Abstract"])]
        survey_proba = predict_survey_proba(texts, tokenizer, model, batch_size=batch_size,
                                            max_length=config.get("max_length", 384),
                                            show_progress=show_progress)

    if mode == "model":
        return (survey_proba >= threshold).astype(int), survey_proba
    if mode == "or":
        return or_rule(survey_proba, frame["Title"], threshold), survey_proba

    if mode == "learned":
        profile_combiner = SurveyCombiner.load(model_path)
        if profile_combiner is not None:
            X = combiner_features(frame, survey_proba, cls_embeddings(frame, model_path, batch_size))
            scores = profile_combiner.predict_proba(X)
            return (scores >= profile_combiner.threshold).astype(int), scores

    # Combiner fitted on the validation split only (earlier learned hybrid)
    combiner = LearnedHybrid.load(model_path)
    if combiner is None:
        print("⚠️  No hybrid combiner found for this model; falling back to the OR rule")
        return or_rule(survey_proba, frame["Title"], threshold), survey_proba
    scores = combiner.predict_proba(frame, survey_proba, use_refs=use_refs)
    return (scores >= combiner.threshold).astype(int), scores


def is_non_paper(frame: pd.DataFrame) -> np.ndarray:
    """Books, editorials, errata, ...: by publication type when known, otherwise by title."""
    by_title = frame["Title"].map(lambda t: bool(NON_PAPER_TITLE.search(clean_text(t))))
    by_type = frame["Type"].map(lambda t: bool(NON_PAPER_TYPES.search(t)))
    return (by_title | by_type).values


def is_magazine_without_survey_signal(frame: pd.DataFrame) -> np.ndarray:
    """
    Magazine articles that do not explicitly present themselves as a survey: no survey term
    in the title, no survey phrasing in the abstract, and not typed 'Review' by the indexer.
    """
    magazine = frame["Venue"].str.strip().map(lambda v: bool(MAGAZINE_VENUE.search(v)))
    explicit = (frame["Title"].map(lambda t: bool(TITLE_SURVEY_TERMS.search(clean_text(t)))) |
                frame["Abstract"].map(lambda a: bool(EXPLICIT_SURVEY_CUES.search(clean_text(a)))) |
                frame["Type"].str.contains(r"\breview\b", case=False, regex=True))
    return (magazine & ~explicit).values


def has_no_metadata(frame: pd.DataFrame) -> np.ndarray:
    """Only a title is known (no abstract, venue or publication type): typically books and chapters."""
    no_abstract = frame["Abstract"].map(lambda a: len(clean_text(a).split()) < 10)
    return (no_abstract & (frame["Venue"].str.strip() == "") & (frame["Type"].str.strip() == "")).values


def categorize(frame: pd.DataFrame, is_survey: np.ndarray, magazine_rule: bool = True) -> np.ndarray:
    """Return one of: non-paper, survey, magazine-overview, research."""
    # With only a title, book titles look like overviews; require an explicit survey keyword instead
    title_only = has_no_metadata(frame)
    keyword = np.array([keyword_is_survey(t) for t in frame["Title"]], dtype=int)
    is_survey = np.where(title_only, keyword, is_survey)

    non_paper = is_non_paper(frame)
    magazine_only = is_magazine_without_survey_signal(frame) if magazine_rule else np.zeros(len(frame), bool)
    return np.where(non_paper, "non-paper",
           np.where(is_survey == 1, np.where(magazine_only, "magazine-overview", "survey"), "research"))


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

def exclude_predicted_surveys(
    input_csv: str,
    output_csv: str,
    survey_csv: str = "Survey-Papers.csv",
    model_path: str = "./distilbert_survey_model",
    threshold: Optional[float] = None,
    mode: str = "learned",
    batch_size: int = 32,
    exclude_magazine_overviews: bool = False,
    magazine_rule: Optional[bool] = None,
) -> dict:
    """
    Classify papers, save research / excluded papers, and print metric changes.
    magazine_rule: report magazine articles without survey framing separately (default: off for the
    learned hybrid, which is fitted on labeled magazine articles; on for the other modes).
    """
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required_cols = {"title", "abstract", "citationCount"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"CSV must contain: {required_cols}")
    df["citationCount"] = df["citationCount"].fillna(0).astype(int)
    print(f"✅ Loaded {len(df)} papers from '{input_csv}'")
    print(f"📝 Mode: {mode}")

    frame = prepare_frame(df)
    is_survey, score = classify_frame(frame, model_path=model_path, mode=mode, threshold=threshold,
                                      batch_size=batch_size, show_progress=True)
    if magazine_rule is None:
        magazine_rule = DEFAULT_MAGAZINE_RULE if mode == "learned" and SurveyCombiner.load(model_path) else True
    df["Category"] = categorize(frame, is_survey, magazine_rule=magazine_rule)
    df["SurveyScore"] = np.round(score, 4)
    excluded_categories = {"survey"} | ({"magazine-overview"} if exclude_magazine_overviews else set())
    df["Prediction"] = (~df["Category"].isin(excluded_categories)).astype(int)  # 0 = survey, 1 = non-survey

    # Only surveys are removed from the metrics; non-papers stay in both profiles
    excluded_df = df[df["Prediction"] == 0]
    filtered_df = df[df["Prediction"] == 1]
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
    counts = df["Category"].value_counts()
    n_surveys, n_non_papers = int(counts.get("survey", 0)), int(counts.get("non-paper", 0))
    n_magazine = int(counts.get("magazine-overview", 0))

    print(f"\n📊 Papers excluded: {len(excluded_df)} ({100 * len(excluded_df) / max(1, total_papers):.2f}%)"
          f" — {n_surveys} surveys"
          + (f", {n_magazine} magazine overviews" if exclude_magazine_overviews else ""))
    if not exclude_magazine_overviews and n_magazine:
        print(f"ℹ️  {n_magazine} magazine articles without explicit survey framing were kept "
              f"(use --exclude-magazine-overviews to drop them)")
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
        "papers": total_papers, "excluded": len(excluded_df), "surveys": n_surveys,
        "non_papers": n_non_papers, "magazine_overviews": n_magazine,
        "citations": total_citations, "excluded_citations": excluded_citations,
        "h_before": total_h, "h_after": filtered_h, "i10_before": total_i10, "i10_after": filtered_i10,
    }


def run_classification_pipeline(
    input_csv: str,
    output_csv: str,
    survey_csv: str,
    model_path: str,
    batch_size: int = 32,
    threshold: Optional[float] = None,
    mode: str = "learned",
    exclude_magazine_overviews: bool = False,
    magazine_rule: Optional[bool] = None,
) -> dict:
    return exclude_predicted_surveys(
        input_csv=input_csv,
        output_csv=output_csv,
        survey_csv=survey_csv,
        model_path=model_path,
        threshold=threshold,
        mode=mode,
        batch_size=batch_size,
        exclude_magazine_overviews=exclude_magazine_overviews,
        magazine_rule=magazine_rule,
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Exclude surveys and recalculate author metrics")
    parser.add_argument("--input", type=str, default="./data/proauthor/auth1.csv")
    parser.add_argument("--output", type=str, default="data/Non-Survey-Papers.csv")
    parser.add_argument("--surveys", type=str, default="data/Survey-Papers.csv")
    parser.add_argument("--model", type=str, default="./distilbert_survey_model")
    parser.add_argument("--mode", type=str, default="learned", choices=MODES)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--exclude-magazine-overviews", action="store_true")
    parser.add_argument("--magazine-rule", action="store_true", default=None)
    args = parser.parse_args()

    run_classification_pipeline(args.input, args.output, args.surveys, args.model,
                                threshold=args.threshold, mode=args.mode,
                                exclude_magazine_overviews=args.exclude_magazine_overviews,
                                magazine_rule=args.magazine_rule)
