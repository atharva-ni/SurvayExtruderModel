"""
Real Training Dataset Builder (OpenAlex)
========================================
Builds a labeled survey / non-survey dataset from real publications.

Labels come from the publication venue (distant supervision):
  * Survey (0)     - survey-only journals (IEEE COMST, ACM CSUR, ...)
  * Non-survey (1) - primary research journals from the SAME topic group

Non-surveys are sampled per topic group and matched to the survey year
distribution, so topic and year cannot be used as shortcuts by the model.
Papers from research venues that look like surveys (survey terms in the title,
or typed as 'review' by OpenAlex) are NOT used for training; they are written
to a separate file as hard cases for manual labeling.
"""

import os
import re
import json
import time
import random
import argparse
from collections import Counter
from typing import Dict, List, Optional

import pandas as pd

import openalex

MAILTO = openalex.MAILTO

# ============================================================================
# VENUES (OpenAlex source IDs, verified by ISSN)
# ============================================================================

SURVEY_VENUES = {
    "S23688054":   ("IEEE Communications Surveys & Tutorials", "comm"),   # 1553-877X
    "S157921468":  ("ACM Computing Surveys", "cs"),                       # 0360-0300
    "S122814990":  ("Artificial Intelligence Review", "cs"),              # 0269-2821
    "S73121659":   ("Computer Science Review", "cs"),                     # 1574-0137
}

RESEARCH_VENUES = {
    # Communications / networking - topic match for COMST
    "S90422530":   ("IEEE Journal on Selected Areas in Communications", "comm"),
    "S63459445":   ("IEEE Transactions on Wireless Communications", "comm"),
    "S62238642":   ("IEEE/ACM Transactions on Networking", "comm"),
    "S196647941":  ("IEEE Transactions on Communications", "comm"),
    "S69141925":   ("IEEE Transactions on Mobile Computing", "comm"),
    "S2480266640": ("IEEE Internet of Things Journal", "comm"),
    "S147316732":  ("IEEE Communications Letters", "comm"),              # short-format research
    "S2500830676": ("IEEE Wireless Communications Letters", "comm"),
    # General computer science - topic match for CSUR / AI Review / CS Review
    "S199944782":  ("IEEE Transactions on Pattern Analysis and Machine Intelligence", "cs"),
    "S4210175523": ("IEEE Transactions on Neural Networks and Learning Systems", "cs"),
    "S30698027":   ("IEEE Transactions on Knowledge and Data Engineering", "cs"),
    "S8351582":    ("IEEE Transactions on Software Engineering", "cs"),
    "S142627899":  ("ACM Transactions on Software Engineering and Methodology", "cs"),
    "S61310614":   ("IEEE Transactions on Information Forensics and Security", "cs"),
    "S118992489":  ("Journal of the ACM", "cs"),
    "S196139623":  ("Artificial Intelligence", "cs"),
    "S120629676":  ("IEEE Signal Processing Letters", "cs"),            # short-format research
    "S151820558":  ("Pattern Recognition Letters", "cs"),
    "S147953040":  ("Information Processing Letters", "cs"),
}

# ============================================================================
# FILTERS
# ============================================================================

# Front matter, errata, etc. that survey venues also publish
JUNK_TITLE = re.compile(
    r"\b(editorial|erratum|errata|corrigendum|correction to|corrections to|in memoriam|"
    r"introduction to the special|special issue|reviewers|table of contents|front cover|"
    r"call for papers|editor'?s note|message from|retraction)\b",
    re.IGNORECASE,
)

# Survey-like titles in research venues -> hard cases, not training data
SURVEY_TITLE = re.compile(
    r"\b(survey|surveys|review|reviews|tutorial|overview|taxonomy|state[- ]of[- ]the[- ]art|"
    r"systematic mapping|primer|roadmap|comparative study|literature)\b",
    re.IGNORECASE,
)

SURVEY_TYPES = {"article", "review"}
RESEARCH_TYPES = {"article"}
MIN_ABSTRACT_WORDS = 50

SELECT_FIELDS = "id,doi,title,publication_year,abstract_inverted_index,cited_by_count,referenced_works_count,type"


# ============================================================================
# OPENALEX CLIENT
# ============================================================================

def _get(params: dict) -> dict:
    return openalex.get("works", params)


def fetch_all(source_id: str, years: str) -> List[dict]:
    """Fetch every work with an abstract from a venue (cursor paging)."""
    works, cursor = [], "*"
    base = {
        "filter": f"primary_location.source.id:{source_id},publication_year:{years},has_abstract:true",
        "select": SELECT_FIELDS,
        "per-page": 200,
    }
    while cursor:
        data = _get({**base, "cursor": cursor})
        works.extend(data.get("results", []))
        cursor = data.get("meta", {}).get("next_cursor")
        if not data.get("results"):
            break
        time.sleep(0.1)
    return works


def fetch_sample(source_id: str, years: str, n: int, seed: int) -> List[dict]:
    """Fetch a random sample of up to n works with an abstract from a venue."""
    works, page = [], 1
    base = {
        "filter": f"primary_location.source.id:{source_id},publication_year:{years},has_abstract:true,type:article",
        "select": SELECT_FIELDS,
        "per-page": 200,
        "sample": n,
        "seed": seed,
    }
    while len(works) < n:
        data = _get({**base, "page": page})
        results = data.get("results", [])
        if not results:
            break
        works.extend(results)
        page += 1
        time.sleep(0.1)
    return works[:n]


def cached(cache_dir: str, key: str, fn):
    path = os.path.join(cache_dir, f"{key}.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    works = fn()
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(works, f)
    return works


# ============================================================================
# CLEANING
# ============================================================================

def rebuild_abstract(inverted: Optional[Dict[str, List[int]]]) -> str:
    if not inverted:
        return ""
    positions = [(pos, word) for word, idxs in inverted.items() for pos in idxs]
    return " ".join(word for _, word in sorted(positions))


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")                  # JATS / HTML tags
    text = re.sub(r"^\s*abstract[:.\s-]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(©|\(c\)|copyright).*$", "", text, flags=re.IGNORECASE)  # trailing copyright
    return re.sub(r"\s+", " ", text).strip()


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())


def to_row(w: dict, venue: str, group: str, label: int) -> dict:
    return {
        "Title": clean_text(w.get("title") or ""),
        "Abstract": clean_text(rebuild_abstract(w.get("abstract_inverted_index"))),
        "Label": label,
        "Venue": venue,
        "Group": group,
        "Year": w.get("publication_year"),
        "Citation": w.get("cited_by_count"),
        "ReferenceCount": w.get("referenced_works_count"),
        "Type": w.get("type"),
        "OpenAlexId": (w.get("id") or "").split("/")[-1],
        "DOI": w.get("doi"),
    }


def usable(row: dict) -> bool:
    return (
        bool(row["Title"])
        and len(row["Abstract"].split()) >= MIN_ABSTRACT_WORDS
        and not JUNK_TITLE.search(row["Title"])
    )


# ============================================================================
# YEAR-MATCHED SAMPLING
# ============================================================================

def year_matched_sample(pool: pd.DataFrame, target_years: Counter, seed: int) -> pd.DataFrame:
    """Sample from pool so its year histogram matches target_years."""
    rng = random.Random(seed)
    chosen, deficit = [], 0
    by_year = {y: list(g.index) for y, g in pool.groupby("Year")}

    for year, need in target_years.items():
        idx = by_year.get(year, [])
        rng.shuffle(idx)
        chosen.extend(idx[:need])
        deficit += max(0, need - len(idx))
        by_year[year] = idx[need:]

    # Fill any shortfall from the remaining pool, preferring the closest years
    if deficit:
        target_mean = sum(y * c for y, c in target_years.items()) / max(1, sum(target_years.values()))
        leftovers = [i for idxs in by_year.values() for i in idxs]
        leftovers.sort(key=lambda i: abs(pool.at[i, "Year"] - target_mean))
        chosen.extend(leftovers[:deficit])

    return pool.loc[chosen]


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_dataset(
    output_csv: str = "data/real_dataset.csv",
    hard_cases_csv: str = "data/hard_cases_to_label.csv",
    years: str = "2010-2025",
    pool_per_venue: int = 1200,
    cache_dir: str = "data/cache/openalex",
    seed: int = 42,
    exclude_csv: Optional[str] = "data/eval_to_label.csv",
) -> pd.DataFrame:
    print("📚 Building real survey/non-survey dataset from OpenAlex")
    if not openalex.API_KEY:
        print("   ℹ️  Tip: set OPENALEX_API_KEY in .env (free at openalex.org); the keyless daily budget is small")

    # ---- Surveys ----
    survey_rows = []
    for sid, (venue, group) in SURVEY_VENUES.items():
        works = cached(cache_dir, f"{sid}_{years}_all", lambda: fetch_all(sid, years))
        rows = [to_row(w, venue, group, 0) for w in works if w.get("type") in SURVEY_TYPES]
        rows = [r for r in rows if usable(r)]
        print(f"   🟢 {venue}: {len(works)} fetched → {len(rows)} usable surveys")
        survey_rows.extend(rows)

    # ---- Research papers ----
    research_rows, hard_rows = [], []
    for sid, (venue, group) in RESEARCH_VENUES.items():
        works = cached(
            cache_dir, f"{sid}_{years}_sample{pool_per_venue}_seed{seed}",
            lambda: fetch_sample(sid, years, pool_per_venue, seed),
        )
        kept = hard = 0
        for w in works:
            row = to_row(w, venue, group, 1)
            if not usable(row):
                continue
            if w.get("type") not in RESEARCH_TYPES or SURVEY_TITLE.search(row["Title"]):
                row["Label"] = None  # unknown: needs a human
                hard_rows.append(row)
                hard += 1
            else:
                research_rows.append(row)
                kept += 1
        print(f"   🔵 {venue}: {len(works)} fetched → {kept} usable, {hard} set aside as hard cases")

    surveys = pd.DataFrame(survey_rows)
    research = pd.DataFrame(research_rows)

    # ---- Deduplicate (within and across classes) ----
    surveys["_key"] = surveys["Title"].map(norm_title)
    research["_key"] = research["Title"].map(norm_title)
    surveys = surveys.drop_duplicates("_key")
    research = research.drop_duplicates("_key")
    clash = set(surveys["_key"]) & set(research["_key"])
    surveys = surveys[~surveys["_key"].isin(clash)]
    research = research[~research["_key"].isin(clash)].reset_index(drop=True)

    # ---- Keep the hand-labeled test set out of training ----
    if exclude_csv and os.path.exists(exclude_csv):
        excluded = pd.read_csv(exclude_csv)
        title_col = "Title" if "Title" in excluded.columns else "title"
        test_keys = set(excluded[title_col].map(norm_title))
        n_before = len(surveys) + len(research)
        surveys = surveys[~surveys["_key"].isin(test_keys)]
        research = research[~research["_key"].isin(test_keys)].reset_index(drop=True)
        print(f"   🧹 Removed {n_before - len(surveys) - len(research)} papers that are in '{exclude_csv}'")

    # ---- Topic- and year-matched negatives ----
    parts = []
    for group, g_surveys in surveys.groupby("Group"):
        target = Counter(g_surveys["Year"])
        pool = research[research["Group"] == group]
        if len(pool) < len(g_surveys):
            print(f"   ⚠️  Group '{group}': only {len(pool)} research papers for {len(g_surveys)} surveys "
                  f"(increase --pool-per-venue)")
        parts.append(year_matched_sample(pool, target, seed))
    matched_research = pd.concat(parts)

    dataset = (
        pd.concat([surveys, matched_research])
        .drop(columns="_key")
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    # ---- Save ----
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    dataset.to_csv(output_csv, index=False)

    hard_df = pd.DataFrame(hard_rows)
    if not hard_df.empty:
        hard_df = hard_df.drop_duplicates(subset="Title")
        hard_df.to_csv(hard_cases_csv, index=False)

    # ---- Summary ----
    print(f"\n✅ Saved {len(dataset)} papers to '{output_csv}'")
    print(dataset.groupby(["Group", "Label"]).size().unstack().rename(columns={0: "survey", 1: "non-survey"}))
    print("\nPapers per venue:")
    print(dataset["Venue"].value_counts().to_string())
    if not hard_df.empty:
        print(f"\n📝 {len(hard_df)} hard cases saved to '{hard_cases_csv}' (fill the Label column by hand)")
    return dataset


def run_build(output_csv=None, hard_cases_csv=None, years="2010-2025", pool_per_venue=1200, seed=42,
              exclude_csv="data/eval_to_label.csv"):
    build_dataset(
        output_csv=output_csv or "data/real_dataset.csv",
        hard_cases_csv=hard_cases_csv or "data/hard_cases_to_label.csv",
        years=years,
        pool_per_venue=pool_per_venue,
        seed=seed,
        exclude_csv=exclude_csv,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a real survey/non-survey dataset from OpenAlex")
    parser.add_argument("--output", type=str, default="data/real_dataset.csv")
    parser.add_argument("--hard-cases", type=str, default="data/hard_cases_to_label.csv")
    parser.add_argument("--years", type=str, default="2010-2025")
    parser.add_argument("--pool-per-venue", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", type=str, default="data/eval_to_label.csv",
                        help="CSV of test papers (by title) to keep out of the dataset")
    args = parser.parse_args()
    run_build(args.output, args.hard_cases, args.years, args.pool_per_venue, args.seed, args.exclude)
