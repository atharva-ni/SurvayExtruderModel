"""
External Test Set: Kaggle arXiv papers from unseen venues
=========================================================
Builds a survey / non-survey test set whose text comes from the Kaggle arXiv
metadata snapshot (Cornell-University/arxiv) and whose labels come from where
each paper was later published. No venue here is used in training:

  * Survey (0)     - survey-only venues (Foundations and Trends, Annual Reviews,
                     Statistics Surveys, IEEE Reviews in Biomedical Engineering, ...)
  * Non-survey (1) - research journals from the same topic group (JMLR, IEEE TIT,
                     IEEE TAC, Theoretical Computer Science, ...)

Published papers are fetched from OpenAlex and matched to arXiv records by
normalized title; title and abstract are the arXiv (Kaggle) versions, the
reference count comes from OpenAlex. Cleaning follows build_dataset.py:
front matter is dropped, research-venue papers that look like surveys are set
aside, research papers are year-matched to the surveys within each topic group,
and any paper in the training or author-profile evaluation sets is removed.
"""

import os
import re
import json
import argparse
from collections import Counter

import pandas as pd

os.environ.setdefault("KAGGLEHUB_CACHE", os.path.join("data", "cache", "kaggle"))

import openalex
from build_dataset import (JUNK_TITLE, SURVEY_TITLE, SURVEY_TYPES, RESEARCH_TYPES, MIN_ABSTRACT_WORDS,
                           cached, clean_text, norm_title, year_matched_sample)

KAGGLE_DATASET = "Cornell-University/arxiv"
KAGGLE_FILE = "arxiv-metadata-oai-snapshot.json"

# Survey-only venues (OpenAlex source IDs); none of them is a training venue
SURVEY_VENUES = {
    "S4210188176": ("Foundations and Trends in Machine Learning", "ml"),
    "S4210187074": ("Foundations and Trends in Optimization", "ml"),
    "S83210289":   ("Foundations and Trends in Computer Graphics and Vision", "ml"),
    "S197106261":  ("Foundations and Trends in Information Retrieval", "ml"),
    "S139685423":  ("Foundations and Trends in Databases", "ml"),
    "S189699570":  ("Foundations and Trends in Human-Computer Interaction", "ml"),
    "S73733540":   ("Foundations and Trends in Web Science", "ml"),
    "S2505707916": ("WIREs Data Mining and Knowledge Discovery", "ml"),
    "S65486112":   ("Archives of Computational Methods in Engineering", "ml"),
    "S123473935":  ("Foundations and Trends in Signal Processing", "comm"),
    "S70831867":   ("Foundations and Trends in Communications and Information Theory", "comm"),
    "S23590115":   ("Foundations and Trends in Networking", "comm"),
    "S4210223006": ("Foundations and Trends in Privacy and Security", "comm"),
    "S4210212775": ("Foundations and Trends in Electric Energy Systems", "comm"),
    "S43319301":   ("Foundations and Trends in Robotics", "control"),
    "S4210220164": ("Foundations and Trends in Systems and Control", "control"),
    "S4210191328": ("Annual Review of Control, Robotics, and Autonomous Systems", "control"),
    "S54761077":   ("Annual Reviews in Control", "control"),
    "S4210217375": ("Annual Review of Statistics and Its Application", "stats"),
    "S147498506":  ("Statistics Surveys", "stats"),
    "S110848248":  ("Foundations and Trends in Theoretical Computer Science", "theory"),
    "S4210233022": ("Foundations and Trends in Programming Languages", "theory"),
    "S138962964":  ("Foundations and Trends in Electronic Design Automation", "theory"),
    "S203619773":  ("IEEE Reviews in Biomedical Engineering", "biomed"),
    "S4210232080": ("Annual Review of Biomedical Data Science", "biomed"),
}

# Research journals from the same topic groups; none of them is a training venue
RESEARCH_VENUES = {
    "S118988714":  ("Journal of Machine Learning Research", "ml"),
    "S207023548":  ("Neural Computation", "ml"),
    "S62148650":   ("Machine Learning", "ml"),
    "S4210173141": ("IEEE Transactions on Image Processing", "ml"),
    "S45693802":   ("Neurocomputing", "ml"),
    "S2729999759": ("Transactions of the Association for Computational Linguistics", "ml"),
    "S79460864":   ("Information Retrieval", "ml"),
    "S4502562":    ("IEEE Transactions on Information Theory", "comm"),
    "S168680287":  ("IEEE Transactions on Signal Processing", "comm"),
    "S10936095":   ("IEEE Transactions on Vehicular Technology", "comm"),
    "S63392143":   ("Computer Networks", "comm"),
    "S184954342":  ("IEEE Transactions on Automatic Control", "control"),
    "S51360982":   ("Automatica", "control"),
    "S144620930":  ("IEEE Transactions on Robotics", "control"),
    "S2502544478": ("IEEE Transactions on Control of Network Systems", "control"),
    "S4394736638": ("Journal of the American Statistical Association", "stats"),
    "S90727058":   ("Theoretical Computer Science", "theory"),
    "S153560523":  ("SIAM Journal on Computing", "theory"),
    "S2495854775": ("IEEE Journal of Biomedical and Health Informatics", "biomed"),
    "S58069681":   ("IEEE Transactions on Medical Imaging", "biomed"),
    "S4210233665": ("IEEE Transactions on Computational Imaging", "biomed"),
}

SELECT_FIELDS = "id,doi,title,publication_year,referenced_works_count,type"


def fetch_all(source_id: str, years: str) -> list:
    return openalex.get_all("works", {"filter": f"primary_location.source.id:{source_id},publication_year:{years}",
                                      "select": SELECT_FIELDS})


def fetch_sample(source_id: str, years: str, n: int, seed: int) -> list:
    works, page = [], 1
    base = {"filter": f"primary_location.source.id:{source_id},publication_year:{years}",
            "select": SELECT_FIELDS, "per-page": 200, "sample": n, "seed": seed}
    while len(works) < n:
        results = openalex.get("works", {**base, "page": page}).get("results", [])
        if not results:
            break
        works.extend(results)
        page += 1
    return works[:n]


def kaggle_snapshot_path() -> str:
    import kagglehub  # public dataset; no Kaggle credentials needed
    return os.path.join(kagglehub.dataset_download(KAGGLE_DATASET), KAGGLE_FILE)


def arxiv_records(snapshot: str, wanted: set) -> dict:
    """{normalized title: arXiv record} for the wanted titles (first arXiv record wins)."""
    found = {}
    with open(snapshot, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = norm_title(rec.get("title"))
            if key in wanted and key not in found:
                found[key] = rec
    return found


def to_row(w: dict, rec: dict, venue: str, group: str, label) -> dict:
    return {
        "Title": clean_text(re.sub(r"\s+", " ", rec.get("title") or "")),
        "Abstract": clean_text(re.sub(r"\s+", " ", rec.get("abstract") or "")),
        "Label": label,
        "Venue": venue,
        "Group": group,
        "Year": w.get("publication_year"),
        "ReferenceCount": w.get("referenced_works_count"),
        "Type": w.get("type"),
        "ArxivId": rec.get("id"),
        "ArxivCategories": rec.get("categories"),
        "OpenAlexId": (w.get("id") or "").split("/")[-1],
        "DOI": w.get("doi"),
    }


def build_kaggle_test(
    output_csv: str = "data/kaggle_arxiv_test.csv",
    years: str = "2010-2025",
    pool_per_venue: int = 3000,
    cache_dir: str = "data/cache/openalex",
    seed: int = 42,
    exclude_csvs=("data/real_dataset.csv", "data/eval_to_label.csv"),
) -> pd.DataFrame:
    print("📚 Building the external test set (Kaggle arXiv text, labels from unseen publication venues)")

    works = []
    for sid, (venue, group) in SURVEY_VENUES.items():
        for w in cached(cache_dir, f"ext_{sid}_{years}_all", lambda: fetch_all(sid, years)):
            works.append((w, venue, group, 0))
    for sid, (venue, group) in RESEARCH_VENUES.items():
        for w in cached(cache_dir, f"ext_{sid}_{years}_sample{pool_per_venue}_seed{seed}",
                        lambda: fetch_sample(sid, years, pool_per_venue, seed)):
            works.append((w, venue, group, 1))
    print(f"   🌐 {len(works)} published papers from {len(SURVEY_VENUES)} survey and "
          f"{len(RESEARCH_VENUES)} research venues")

    snapshot = kaggle_snapshot_path()
    print(f"   📦 Matching titles against '{snapshot}' (takes a few minutes)...")
    records = arxiv_records(snapshot, {norm_title(w.get("title")) for w, *_ in works} - {""})

    survey_rows, research_rows = [], []
    stats = Counter()
    for w, venue, group, label in works:
        rec = records.get(norm_title(w.get("title")))
        if rec is None:
            continue
        row = to_row(w, rec, venue, group, label)
        if len(row["Abstract"].split()) < MIN_ABSTRACT_WORDS or JUNK_TITLE.search(row["Title"]):
            stats["unusable"] += 1
        elif label == 0:
            if w.get("type") in SURVEY_TYPES:
                survey_rows.append(row)
            else:
                stats["survey venue, not an article"] += 1
        elif w.get("type") not in RESEARCH_TYPES or SURVEY_TITLE.search(row["Title"]):
            stats["research venue, looks like a survey (set aside)"] += 1
        else:
            research_rows.append(row)

    surveys = pd.DataFrame(survey_rows)
    research = pd.DataFrame(research_rows)
    surveys["_key"] = surveys["Title"].map(norm_title)
    research["_key"] = research["Title"].map(norm_title)
    surveys = surveys.drop_duplicates("_key")
    research = research.drop_duplicates("_key")
    clash = set(surveys["_key"]) & set(research["_key"])
    surveys = surveys[~surveys["_key"].isin(clash)]
    research = research[~research["_key"].isin(clash)].reset_index(drop=True)

    # Nothing the model was trained on, and nothing from the author-profile test set
    seen = set()
    for path in exclude_csvs:
        if path and os.path.exists(path):
            df = pd.read_csv(path)
            seen |= set(df["Title" if "Title" in df.columns else "title"].map(norm_title))
    n_before = len(surveys) + len(research)
    surveys = surveys[~surveys["_key"].isin(seen)]
    research = research[~research["_key"].isin(seen)].reset_index(drop=True)
    stats["already in training / evaluation sets"] = n_before - len(surveys) - len(research)

    parts = []
    for group, g_surveys in surveys.groupby("Group"):
        pool = research[research["Group"] == group]
        if len(pool) < len(g_surveys):
            print(f"   ⚠️  Group '{group}': only {len(pool)} research papers for {len(g_surveys)} surveys")
        parts.append(year_matched_sample(pool, Counter(g_surveys["Year"]), seed))

    dataset = (pd.concat([surveys] + parts).drop(columns="_key")
               .sample(frac=1, random_state=seed).reset_index(drop=True))
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    dataset.to_csv(output_csv, index=False)

    print(f"   🔗 Matched to arXiv: {len(records)} titles")
    for reason, n in stats.items():
        print(f"   🧹 {reason}: {n}")
    print(f"\n✅ Saved {len(dataset)} papers to '{output_csv}'")
    print(dataset.groupby(["Group", "Label"]).size().unstack().rename(columns={0: "survey", 1: "non-survey"}))
    print("\nSurveys per venue:")
    print(dataset[dataset["Label"] == 0]["Venue"].value_counts().to_string())
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data/kaggle_arxiv_test.csv")
    parser.add_argument("--years", default="2010-2025")
    parser.add_argument("--pool-per-venue", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_kaggle_test(args.output, args.years, args.pool_per_venue, seed=args.seed)
