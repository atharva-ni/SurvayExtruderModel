"""
Combine Author Profiles from Several Sources
============================================
Semantic Scholar and OpenAlex each miss part of a prolific author's work (split
or conflated author IDs). This merges extraction CSVs into one profile:
  * one row per normalized title (versions of the same work are merged),
  * citationCount / ReferenceCount: the highest value across sources,
  * abstract: the longest available; venue / year: first non-empty,
  * type: all publication types seen (so a 'book' tag from any source counts),
  * Sources: which inputs contained the work.
"""

import os
import re
import argparse
from typing import Dict, List

import pandas as pd


def norm_title(title) -> str:
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def load(path: str, source: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame({
        "title": df["title"].fillna("").astype(str),
        "abstract": df["abstract"].fillna("").astype(str),
        "authors": df.get("authors", pd.Series("", index=df.index)).fillna("").astype(str),
        "citationCount": pd.to_numeric(df["citationCount"], errors="coerce").fillna(0).astype(int),
        "venue": df.get("venue", pd.Series("", index=df.index)).fillna("").astype(str),
        "year": pd.to_numeric(df.get("year", pd.Series(None, index=df.index)), errors="coerce"),
        "type": df.get("type", pd.Series("", index=df.index)).fillna("").astype(str),
    })
    if "ReferenceCount" in df.columns:
        out["ReferenceCount"] = pd.to_numeric(df["ReferenceCount"], errors="coerce").fillna(0).astype(int)
    elif "references" in df.columns:
        out["ReferenceCount"] = df["references"].fillna("").map(lambda s: len([r for r in str(s).split(";") if r.strip()]))
    else:
        out["ReferenceCount"] = 0
    out["Sources"] = source
    out["_key"] = out["title"].map(norm_title)
    return out[out["_key"].str.len() > 0]


def first_non_empty(values: pd.Series):
    for v in values:
        if isinstance(v, str) and v.strip():
            return v
        if not isinstance(v, str) and pd.notna(v):
            return v
    return ""


def combine(inputs: Dict[str, str], output_csv: str) -> pd.DataFrame:
    frames = [load(path, source) for source, path in inputs.items() if os.path.exists(path)]
    both = pd.concat(frames, ignore_index=True)

    # Prefer the most cited version's title/venue/year/authors
    both = both.sort_values("citationCount", ascending=False)
    grouped = both.groupby("_key", sort=False)
    combined = pd.DataFrame({
        "title": grouped["title"].first(),
        "abstract": grouped["abstract"].agg(lambda s: max(s, key=len)),
        "authors": grouped["authors"].agg(first_non_empty),
        "citationCount": grouped["citationCount"].max(),
        "ReferenceCount": grouped["ReferenceCount"].max(),
        "venue": grouped["venue"].agg(first_non_empty),
        "year": grouped["year"].agg(first_non_empty),
        "type": grouped["type"].agg(lambda s: "; ".join(sorted({t for v in s for t in re.split(r";\s*", v) if t}))),
        "Sources": grouped["Sources"].agg(lambda s: ", ".join(sorted(set(s)))),
    }).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    combined.to_csv(output_csv, index=False)
    print(f"✅ {len(both)} rows from {len(frames)} source(s) → {len(combined)} works; saved to '{output_csv}'")
    print(f"   {combined['Sources'].value_counts().to_dict()}")
    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="source=path pairs, e.g. semantic=data/a.csv openalex=data/b.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    combine(dict(item.split("=", 1) for item in args.inputs), args.output)
