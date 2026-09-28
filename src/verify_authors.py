"""
Verify author profiles and classifications against Google Scholar
=================================================================
For each author in data/scholar_reference.json:
  1. Profile completeness: papers, citations, h-index and i10-index of each
     extracted profile vs. Google Scholar, and how many of the author's
     Scholar top-20 items the profile contains.
  2. Classification check: the category the classifier assigns to each of the
     Scholar top-20 items found, against a reference label
     (S survey, M magazine overview, R research, B book). Items in the
     training set are marked and reported separately.
"""

import os
import re
import json
import argparse

import pandas as pd
from tabulate import tabulate

from classifier import prepare_frame, classify_frame, categorize, calculate_indices

# Books only have to be kept out of the survey category (they stay in the profile either way)
EXPECTED = {"S": {"survey"}, "M": {"magazine-overview", "survey"}, "R": {"research", "magazine-overview"},
            "B": {"non-paper", "research"}}
PREFERRED_SOURCE = "Semantic Scholar (merged IDs)"


def norm_title(title) -> str:
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def find(profile: pd.DataFrame, title: str):
    """Exact normalized match, else a profile title that starts with the Scholar title."""
    key = norm_title(title)
    hit = profile[profile["_key"] == key]
    if hit.empty and len(key) >= 20:
        hit = profile[profile["_key"].str.startswith(key)]
    return None if hit.empty else hit.sort_values("citationCount", ascending=False).iloc[0]


def verify(reference_path: str, profiles: dict, model_path: str, report_path: str,
           training_csv: str = "data/real_dataset.csv") -> None:
    with open(reference_path, encoding="utf-8") as f:
        reference = json.load(f)

    # Scholar items the model was trained on are reported but not counted as unseen
    train_keys = set()
    if os.path.exists(training_csv):
        train_keys = set(pd.read_csv(training_csv, usecols=["Title"])["Title"].map(norm_title))
    else:
        print(f"⚠️  Training set '{training_csv}' not found; overlap with training data NOT marked")

    coverage_rows, check_rows, details = [], [], []
    all_correct = all_total = unseen_correct = unseen_total = 0
    for author, paths in profiles.items():
        ref = reference[author]
        coverage_rows.append([author, "Google Scholar", "", f"{ref['citations']:,}", ref["h"], ref["i10"], "20/20"])

        for source, path in paths.items():
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)
            df["citationCount"] = df["citationCount"].fillna(0).astype(int)
            df["_key"] = df["title"].map(norm_title)
            h, i10 = calculate_indices(df)
            found = [find(df, t) for t, _ in ref["top20"]]
            coverage_rows.append([author, source, f"{len(df):,}", f"{int(df['citationCount'].sum()):,}", h, i10,
                                  f"{sum(r is not None for r in found)}/20"])

        # Classification check on the preferred source, else the most complete profile
        best_source, best_path = (PREFERRED_SOURCE, paths[PREFERRED_SOURCE]) if os.path.exists(
            paths.get(PREFERRED_SOURCE, "")) else max(
            ((s, p) for s, p in paths.items() if os.path.exists(p)),
            key=lambda sp: sum(find(pd.read_csv(sp[1]).assign(_key=lambda d: d["title"].map(norm_title),
                                                               citationCount=lambda d: d["citationCount"].fillna(0)),
                                    t) is not None for t, _ in ref["top20"]))
        df = pd.read_csv(best_path)
        df["citationCount"] = df["citationCount"].fillna(0).astype(int)
        df["_key"] = df["title"].map(norm_title)
        frame = prepare_frame(df)
        is_survey, _ = classify_frame(frame, model_path=model_path)
        df["Category"] = categorize(frame, is_survey)

        correct = total = u_correct = u_total = 0
        for title, truth in ref["top20"]:
            in_training = norm_title(title) in train_keys
            row = find(df, title)
            if row is None:
                details.append([author, truth, "not in profile", "", "yes" if in_training else "", title[:70]])
                continue
            ok = row["Category"] in EXPECTED[truth]
            correct += ok
            total += 1
            if not in_training:
                u_correct += ok
                u_total += 1
            details.append([author, truth, row["Category"], "✓" if ok else "✗", "yes" if in_training else "",
                            title[:70]])
        check_rows.append([author, best_source, f"{correct}/{total}", f"{u_correct}/{u_total}"])
        all_correct, all_total = all_correct + correct, all_total + total
        unseen_correct, unseen_total = unseen_correct + u_correct, unseen_total + u_total

    pct = lambda c, t: f"{c}/{t} ({100 * c / max(1, t):.0f}%)"
    check_rows.append(["Total", "", pct(all_correct, all_total), pct(unseen_correct, unseen_total)])

    coverage = tabulate(coverage_rows, headers=["Author", "Source", "Papers", "Citations", "h", "i10", "Scholar top-20 found"],
                        tablefmt="github")
    checks = tabulate(check_rows, headers=["Author", "Profile used", "Correct (all found)", "Correct (not in training)"],
                      tablefmt="github")
    detail = tabulate(details, headers=["Author", "Ref", "Classifier", "", "In training", "Title"], tablefmt="github")

    md = ["# Author verification against Google Scholar", "",
          "Reference labels: S survey/tutorial/overview, M magazine overview (survey or magazine-overview accepted), "
          "R research (research or magazine-overview accepted), B book (must not be counted as a survey).", "",
          "## Profile completeness", "", coverage, "", "## Classification of Scholar top-20 items", "",
          f"Items whose title is in the training set (`{training_csv}`) are counted only in the first column.", "",
          checks, "", "## Details", "", detail, ""]
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(coverage, "\n")
    print(checks, "\n")
    print(detail)
    print(f"\n✅ Report saved to '{report_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", default="data/scholar_reference.json")
    parser.add_argument("--model", default="./distilbert_survey_model")
    parser.add_argument("--report", default="reports/author_verification.md")
    parser.add_argument("--training", default="data/real_dataset.csv", help="Training dataset (items in it are marked)")
    args = parser.parse_args()

    profiles = {
        "hanzo":   {"Semantic Scholar": "data/proauthor/auth1.csv", "Semantic Scholar (merged IDs)": "data/proauthor_s2/hanzo.csv",
                    "OpenAlex": "data/proauthor_openalex/hanzo.csv",
                    "Combined": "data/proauthor_combined/hanzo.csv"},
        "niyato":  {"Semantic Scholar": "data/proauthor/auth2.csv", "Semantic Scholar (merged IDs)": "data/proauthor_s2/niyato.csv",
                    "OpenAlex": "data/proauthor_openalex/niyato.csv", "Combined": "data/proauthor_combined/niyato.csv"},
        "yu":      {"Semantic Scholar": "data/proauthor/auth3.csv", "Semantic Scholar (merged IDs)": "data/proauthor_s2/yu.csv",
                    "OpenAlex": "data/proauthor_openalex/yu.csv",
                    "Combined": "data/proauthor_combined/yu.csv"},
        "han":     {"Semantic Scholar": "data/proauthor/auth4.csv", "Semantic Scholar (merged IDs)": "data/proauthor_s2/han.csv",
                    "OpenAlex": "data/proauthor_openalex/han.csv",
                    "Combined": "data/proauthor_combined/han.csv"},
        "hossain": {"Semantic Scholar": "data/proauthor/auth5.csv", "Semantic Scholar (merged IDs)": "data/proauthor_s2/hossain.csv",
                    "OpenAlex": "data/proauthor_openalex/hossain.csv",
                    "Combined": "data/proauthor_combined/hossain.csv"},
    }
    verify(args.reference, profiles, args.model, args.report, args.training)
