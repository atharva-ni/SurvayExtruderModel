"""
Author Publication Extraction (OpenAlex)
========================================
OpenAlex keeps one merged profile per author, whereas Semantic Scholar often
splits prolific authors across several IDs (e.g. D. Niyato: 1,289 + 703 papers).

Output columns match extract_semantic.py (id, title, abstract, authors,
citationCount, venue, year) plus type, ReferenceCount and doi. Versions of the
same work (e.g. arXiv preprint and journal article) are merged by title,
keeping the published version.
"""

import os
import re
import argparse
from typing import Optional

import pandas as pd

import openalex

WORK_FIELDS = ("id,doi,title,publication_year,type,cited_by_count,referenced_works_count,"
               "abstract_inverted_index,primary_location,authorships")


def find_author(name: str, api_key: Optional[str] = None) -> str:
    """Search by name; return the ID of the candidate with the most works (printed for checking)."""
    results = openalex.get("authors", {"search": name, "per-page": 5}, api_key=api_key).get("results", [])
    if not results:
        raise ValueError(f"No OpenAlex author found for '{name}'")
    print(f"🔎 OpenAlex candidates for '{name}':")
    for a in results:
        inst = (a.get("last_known_institutions") or [{}])[0] or {}
        print(f"   {openalex.short_id(a['id']):12s} {a['display_name'][:30]:30s} works {a['works_count']:5d} "
              f"cites {a['cited_by_count']:7d} | {inst.get('display_name', '')[:40]}")
    best = max(results, key=lambda a: a["works_count"])
    print(f"   → using {openalex.short_id(best['id'])} ({best['display_name']}); pass --author to choose another")
    return openalex.short_id(best["id"])


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def work_to_row(w: dict) -> dict:
    source = ((w.get("primary_location") or {}).get("source") or {})
    return {
        "id": openalex.short_id(w.get("id")),
        "title": w.get("title") or "",
        "abstract": openalex.rebuild_abstract(w.get("abstract_inverted_index")),
        "authors": "; ".join(a.get("author", {}).get("display_name", "") for a in w.get("authorships") or []),
        "citationCount": w.get("cited_by_count") or 0,
        "ReferenceCount": w.get("referenced_works_count") or 0,
        "venue": source.get("display_name") or "",
        "year": w.get("publication_year"),
        "type": w.get("type") or "",
        "doi": w.get("doi") or "",
    }


def merge_versions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per title: prefer the non-preprint version, then the most cited."""
    df = df[df["title"].str.strip() != ""].copy()
    df["_key"] = df["title"].map(norm_title)
    df["_preprint"] = (df["type"] == "preprint").astype(int)
    df = df.sort_values(["_key", "_preprint", "citationCount"], ascending=[True, True, False])
    return df.drop_duplicates("_key").drop(columns=["_key", "_preprint"]).reset_index(drop=True)


def run_extraction(author_id: Optional[str] = None, name: Optional[str] = None,
                   output_path: Optional[str] = None, api_key: Optional[str] = None) -> pd.DataFrame:
    if not author_id and not name:
        raise ValueError("Give an OpenAlex author ID (--author A...) or a name (--name)")
    author_id = author_id or find_author(name, api_key)

    works = openalex.get_all("works", {"filter": f"author.id:{author_id}", "select": WORK_FIELDS}, api_key=api_key)
    df = pd.DataFrame([work_to_row(w) for w in works])
    merged = merge_versions(df)

    output_path = output_path or os.path.join("data", "proauthor_openalex", f"{author_id}.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    merged.to_csv(output_path, index=False)

    print(f"✅ {len(works)} works fetched → {len(merged)} after merging versions; saved to '{output_path}'")
    print(f"   Types: {merged['type'].value_counts().to_dict()}")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract an author's publications from OpenAlex")
    parser.add_argument("--author", type=str, help="OpenAlex author ID, e.g. A5091266202")
    parser.add_argument("--name", type=str, help="Author name to search for")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()
    run_extraction(args.author, args.name, args.output, args.api_key)
