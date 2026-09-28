"""
Comparison cohort for Table V
=============================
Selects prolific communications authors who rarely publish surveys, finds
their Semantic Scholar author IDs, and downloads their merged profiles.

Selection (fixed rule, no hand-picking): among the 60 most prolific authors of
IEEE JSAC and IEEE Transactions on Wireless Communications (2010-2025, OpenAlex),
take the first five, by number of papers there, whose OpenAlex h-index lies in
the range of the survey cohort and who have at most MAX_COMST papers in IEEE
Communications Surveys & Tutorials. The survey-cohort authors are excluded.

Author IDs: Semantic Scholar splits prolific authors across several IDs and
also has namesakes. Candidate IDs come from a name search (every spelling
OpenAlex records) and from the matching author entries on the person's 300
most-cited papers (looked up by DOI). A candidate is kept only if its papers
overlap the author's OpenAlex profile: at least MIN_OVERLAP titles in common
and at least MIN_SHARE of its titles found there.
"""

import os
import re
import json
import time
import argparse

import requests

import openalex
from extract_semantic import API_KEY as S2_KEY

JSAC, TWC, COMST = "S90422530", "S63459445", "S23688054"
SURVEY_COHORT = {"A5091122305": "hanzo", "A5091266202": "niyato", "A5100420016": "yu",
                 "A5063667378": "han", "A5089270885": "hossain"}
N_CANDIDATES, N_COHORT, MAX_COMST = 60, 5, 4
MIN_OVERLAP, MIN_SHARE = 2, 0.5
S2 = "https://api.semanticscholar.org/graph/v1"


def norm_title(t) -> str:
    return re.sub(r"[^a-z0-9]", "", str(t or "").lower())


def oa_count(filt: str) -> int:
    return openalex.get("works", {"filter": filt, "per-page": 1})["meta"]["count"]


def select_cohort() -> tuple:
    h_range = [openalex.get(f"authors/{a}", {"select": "summary_stats"})["summary_stats"]["h_index"]
               for a in SURVEY_COHORT]
    lo, hi = min(h_range), max(h_range)
    groups = openalex.get("works", {"filter": f"primary_location.source.id:{JSAC}|{TWC},publication_year:2010-2025",
                                    "group_by": "authorships.author.id", "per-page": 200})["group_by"]
    candidates, chosen = [], []
    for g in groups[:N_CANDIDATES]:
        aid = openalex.short_id(g["key"])
        if aid in SURVEY_COHORT:
            continue
        a = openalex.get(f"authors/{aid}", {"select": "display_name,summary_stats"})
        row = {"openalex": aid, "name": a["display_name"], "jsac_twc_papers": g["count"],
               "h_index": a["summary_stats"]["h_index"],
               "comst_papers": oa_count(f"author.id:{aid},primary_location.source.id:{COMST}")}
        row["eligible"] = lo <= row["h_index"] <= hi and row["comst_papers"] <= MAX_COMST
        candidates.append(row)
        if row["eligible"] and len(chosen) < N_COHORT:
            chosen.append(row)
    return chosen, candidates, (lo, hi)


def s2_request(method: str, path: str, params: dict, body=None):
    headers = {"x-api-key": S2_KEY.strip()} if S2_KEY else {}
    for attempt in range(8):
        r = requests.request(method, f"{S2}/{path}", params=params, json=body, headers=headers, timeout=120)
        time.sleep(1)
        if r.status_code == 403 and headers:  # revoked key: continue without it
            headers = {}
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * 2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Semantic Scholar request failed: {path}")


def s2_titles(author_ids: list) -> dict:
    """{author id: [normalized titles]} via the batch endpoint (all papers per author)."""
    out = {}
    for start in range(0, len(author_ids), 50):
        chunk = author_ids[start:start + 50]
        for a in s2_request("POST", "author/batch", {"fields": "paperCount,papers.title"}, {"ids": chunk}):
            if a:
                out[a["authorId"]] = [norm_title(p.get("title")) for p in a.get("papers") or []]
    return out


PROFILE_FIELDS = ("papers.paperId,papers.title,papers.abstract,papers.venue,papers.year,papers.citationCount,"
                  "papers.authors,papers.publicationTypes")


def download_profile(author_ids: list, out_csv: str) -> None:
    """Merged profile in the same format as extract_semantic.py, using batch requests (fast without an API key)."""
    papers = {}
    for start in range(0, len(author_ids), 20):
        for a in s2_request("POST", "author/batch", {"fields": PROFILE_FIELDS}, {"ids": author_ids[start:start + 20]}):
            for p in (a or {}).get("papers") or []:
                papers.setdefault(p["paperId"], p)
    # The same work can appear twice (e.g. arXiv and journal versions): keep the most cited, as extract_semantic.py
    by_title = {}
    for p in papers.values():
        key = norm_title(p.get("title"))
        if key and (key not in by_title or (p.get("citationCount") or 0) > (by_title[key].get("citationCount") or 0)):
            by_title[key] = p
    ids = [p["paperId"] for p in by_title.values()]
    refs = {}
    for start in range(0, len(ids), 500):
        for p in s2_request("POST", "paper/batch", {"fields": "references.paperId"}, {"ids": ids[start:start + 500]}):
            if p:
                refs[p["paperId"]] = [r["paperId"] for r in p.get("references") or [] if r.get("paperId")]
    rows = [{
        "id": p["paperId"], "title": p.get("title"), "abstract": p.get("abstract"),
        "authors": "; ".join(a.get("name") or "" for a in p.get("authors") or []),
        "citationCount": p.get("citationCount"), "references": "; ".join(refs.get(p["paperId"], [])),
        "venue": p.get("venue"), "year": p.get("year"), "type": "; ".join(p.get("publicationTypes") or []),
    } for p in by_title.values()]
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"   {len(papers)} papers from {len(author_ids)} IDs, {len(rows)} after merging duplicate titles -> {out_csv}")


def openalex_works(author_id: str) -> list:
    return openalex.get_all("works", {"filter": f"author.id:{author_id}", "select": "title,doi,cited_by_count"})


def openalex_titles(author_id: str) -> set:
    return {norm_title(w.get("title")) for w in openalex_works(author_id)} - {""}


def ids_from_own_papers(works: list, name: str, n_papers: int = 300) -> list:
    """Author IDs on the person's most-cited papers (by DOI) whose name matches: robust for common names."""
    last, first = name.replace("‐", "-").split()[-1].lower(), name[0].lower()
    dois = [w["doi"].replace("https://doi.org/", "") for w in
            sorted(works, key=lambda w: -(w.get("cited_by_count") or 0)) if w.get("doi")][:n_papers]
    found = {}
    for start in range(0, len(dois), 100):
        papers = s2_request("POST", "paper/batch", {"fields": "authors"},
                            {"ids": ["DOI:" + d for d in dois[start:start + 100]]})
        for p in papers:
            for a in (p or {}).get("authors") or []:
                n = (a.get("name") or "").replace("‐", "-").lower()
                if a.get("authorId") and n.split() and n.split()[-1] == last and n[0] == first:
                    found[a["authorId"]] = {"authorId": a["authorId"], "name": a.get("name"), "paperCount": 1}
    return list(found.values())


def name_variants(name: str, openalex_id: str, n: int = 4) -> list:
    """The display name plus OpenAlex's alternative spellings (e.g. 'Shi Hong Jin' is 'Shi Jin' elsewhere)."""
    alts = openalex.get(f"authors/{openalex_id}", {"select": "display_name_alternatives"})
    names = [name] + [a for a in alts.get("display_name_alternatives") or [] if "." not in a and len(a.split()) >= 2]
    seen, out = set(), []
    for v in names:
        key = v.lower().replace("-", " ").replace("‐", " ")
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out[:n]


def resolve_s2_ids(name: str, openalex_id: str) -> dict:
    works = openalex_works(openalex_id)
    reference = {norm_title(w.get("title")) for w in works} - {""}
    hits, seen = [], set()
    # Candidates: name search (each spelling) plus the matching authors on the person's own papers
    searched = [h for variant in name_variants(name, openalex_id)
                for h in s2_request("GET", "author/search", {"query": variant, "fields": "name,paperCount",
                                                             "limit": 100}).get("data", [])]
    for h in searched + ids_from_own_papers(works, name):
        if h.get("paperCount") and h["authorId"] not in seen:
            seen.add(h["authorId"])
            hits.append(h)
    titles_by_id = s2_titles([h["authorId"] for h in hits])
    accepted, rejected = [], []
    for h in hits:
        titles = [t for t in titles_by_id.get(h["authorId"], []) if t]
        common = sum(t in reference for t in titles)
        share = common / max(1, len(titles))
        entry = {"id": h["authorId"], "name": h.get("name"), "papers": len(titles), "in_openalex": common,
                 "share": round(share, 3)}
        (accepted if common >= MIN_OVERLAP and share >= MIN_SHARE else rejected).append(entry)
    return {"openalex_titles": len(reference), "accepted": accepted, "rejected": rejected}


def build_cohort(ids_json="data/cohort_author_ids.json", out_dir="data/proauthor_cohort"):
    chosen, candidates, h_range = select_cohort()
    print(f"h-index range of the survey cohort (OpenAlex): {h_range}")
    previous = {}
    if os.path.exists(ids_json):
        with open(ids_json, encoding="utf-8") as f:
            previous = json.load(f).get("cohort", {})
    record = {"_rule": __doc__.strip().split("\n\n")[1], "h_range": h_range, "max_comst": MAX_COMST,
              "candidates": candidates, "cohort": {}}
    os.makedirs(out_dir, exist_ok=True)
    for row in chosen:
        key = norm_title(row["name"].split()[-1])
        print(f"\n👤 {row['name']} ({row['openalex']}): h={row['h_index']}, COMST papers={row['comst_papers']}")
        ids = resolve_s2_ids(row["name"], row["openalex"])
        accepted = [a["id"] for a in ids["accepted"]]
        print(f"   accepted IDs: {[(a['id'], a['papers'], a['share']) for a in ids['accepted']]}")
        record["cohort"][key] = {**row, "semantic_scholar": accepted, "id_check": ids}
        with open(ids_json, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=1)
        out_csv = os.path.join(out_dir, f"{key}.csv")
        if accepted and os.path.exists(out_csv) and previous.get(key, {}).get("semantic_scholar") == accepted:
            print(f"   profile already downloaded for these IDs: {out_csv}")
        elif accepted:
            download_profile(accepted, out_csv)
    print(f"\n✅ Cohort IDs and selection saved to '{ids_json}'; profiles in '{out_dir}'")


def build_openalex_cohorts(record_json="data/cohort_openalex.json", out_dir="data/cohort_openalex"):
    """Both cohorts from OpenAlex (one merged profile per author), fetched the same day for a like-for-like comparison."""
    from datetime import date
    from extract_openalex import run_extraction as openalex_profile

    chosen, candidates, h_range = select_cohort()
    print(f"h-index range of the survey cohort (OpenAlex): {h_range}")
    record = {"_rule": __doc__.strip().split("\n\n")[1], "source": "OpenAlex", "date": str(date.today()),
              "h_range": h_range, "max_comst": MAX_COMST, "candidates": candidates,
              "comparison": {norm_title(r["name"].split()[-1]): r for r in chosen},
              "survey": {name: {"openalex": aid} for aid, name in SURVEY_COHORT.items()}}
    for group in ("survey", "comparison"):
        for key, row in record[group].items():
            print(f"\n👤 {group}: {key} ({row['openalex']})")
            openalex_profile(author_id=row["openalex"], output_path=os.path.join(out_dir, group, f"{key}.csv"))
    with open(record_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print(f"\n✅ Selection saved to '{record_json}'; profiles in '{out_dir}/survey' and '{out_dir}/comparison'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["openalex", "semantic-scholar"], default="openalex",
                        help="openalex: both cohorts from OpenAlex (default); semantic-scholar: comparison cohort "
                             "from Semantic Scholar with merged author IDs")
    parser.add_argument("--ids", default="data/cohort_author_ids.json")
    parser.add_argument("--out-dir", default="data/proauthor_cohort")
    args = parser.parse_args()
    if args.source == "openalex":
        build_openalex_cohorts()
    else:
        build_cohort(args.ids, args.out_dir)
