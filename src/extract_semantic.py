import os
import time
import csv
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
AUTHOR_ID = os.getenv("SEMANTIC_SCHOLAR_AUTHOR_ID", "40356145")
BASE_URL = "https://api.semanticscholar.org/graph/v1"


def get_all_author_papers(author_id, api_key=None):
    all_papers = []
    offset = 0
    limit = 1000

    # Fallback/override key logic
    key = api_key if api_key is not None else API_KEY
    if key:
        key = key.strip()
    headers = {"x-api-key": key} if key else {}

    while True:
        url = f"{BASE_URL}/author/{author_id}/papers"
        params = {
            "limit": limit,
            "offset": offset,
            "fields": "title,year,venue,abstract,authors,citationCount,references,publicationTypes"
        }

        for attempt in range(8):
            response = requests.get(url, headers=headers, params=params, timeout=120)
            time.sleep(1)  # avoid hitting rate limits
            if response.status_code == 403 and headers:
                print("⚠️  Semantic Scholar rejected the API key (revoked?); continuing without a key")
                headers = {}
                continue
            if response.status_code in (429, 500, 502, 503, 504):
                wait = 5 * 2 ** attempt
                print(f"⏳ Semantic Scholar returned {response.status_code}; retrying in {wait}s...")
                time.sleep(wait)
                continue
            break

        # Never save a silently truncated profile
        if response.status_code != 200:
            raise RuntimeError(f"Semantic Scholar request for author {author_id} failed "
                               f"({response.status_code}): {response.text[:200]}")

        data = response.json().get("data", [])
        if not data:
            print("✅ No more papers found, stopping.")
            break

        all_papers.extend(data)
        print(f"Fetched {len(data)} papers (Total so far: {len(all_papers)})")

        # Stop if less than limit (means last page)
        if len(data) < limit:
            break

        offset += limit  # move to next page

    return all_papers


def extract_paper_info(papers):
    results = []

    for p in papers:
        references_raw = p.get("references", [])
        references_clean = []

        if isinstance(references_raw, list):
            for r in references_raw:
                pid = r.get("paperId")
                if pid:
                    references_clean.append(pid)

        info = {
            "id": p.get("paperId"),
            "title": p.get("title"),
            "abstract": p.get("abstract"),
            "authors": "; ".join([a.get("name") for a in p.get("authors", [])]),
            "citationCount": p.get("citationCount"),
            "references": "; ".join(references_clean),
            "venue": p.get("venue"),
            "year": p.get("year"),
            "type": "; ".join(p.get("publicationTypes") or []),
        }
        results.append(info)

    return results


def save_to_csv(data, filename="data/nima.csv"):
    if not data:
        print("No data to save.")
        return

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ CSV saved successfully as {filename}")


def run_extraction(author_id=None, output_path=None, api_key=None):
    """author_id may list several IDs separated by commas (Semantic Scholar often splits one author)."""
    target_authors = [a.strip() for a in str(author_id or AUTHOR_ID).split(",") if a.strip()]
    target_output = output_path or "data/nima.csv"

    papers, seen = [], set()
    for target in target_authors:
        for p in get_all_author_papers(target, api_key=api_key):
            if p.get("paperId") not in seen:
                seen.add(p.get("paperId"))
                papers.append(p)
    # The same work can appear twice (e.g. arXiv and journal versions): keep the most cited
    by_title = {}
    for p in papers:
        key = "".join(ch for ch in str(p.get("title") or "").lower() if ch.isalnum())
        if key and (key not in by_title or (p.get("citationCount") or 0) > (by_title[key].get("citationCount") or 0)):
            by_title[key] = p
    print(f"Total papers found: {len(papers)} from {len(target_authors)} author ID(s); "
          f"{len(by_title)} after merging duplicate titles")
    papers = list(by_title.values())

    final_data = extract_paper_info(papers)
    save_to_csv(final_data, target_output)


def add_publication_types(csv_path, api_key=None, batch_size=500):
    """Add a 'type' column (Semantic Scholar publicationTypes) to an existing extraction CSV."""
    import pandas as pd

    key = (api_key if api_key is not None else API_KEY or "").strip()
    headers = {"x-api-key": key} if key else {}
    df = pd.read_csv(csv_path)
    ids = df["id"].dropna().astype(str).tolist()
    types = {}

    for start in range(0, len(ids), batch_size):
        chunk = ids[start:start + batch_size]
        for attempt in range(6):
            response = requests.post(f"{BASE_URL}/paper/batch", params={"fields": "publicationTypes"},
                                     json={"ids": chunk}, headers=headers, timeout=120)
            if response.status_code == 200:
                break
            if response.status_code == 403 and headers:
                print("⚠️  Semantic Scholar rejected the API key (revoked?); continuing without a key")
                headers = {}
                continue
            time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Semantic Scholar batch request failed: {response.status_code} {response.text[:200]}")
        for pid, paper in zip(chunk, response.json()):
            types[pid] = "; ".join((paper or {}).get("publicationTypes") or [])
        time.sleep(1)

    df["type"] = df["id"].astype(str).map(types).fillna("")
    df.to_csv(csv_path, index=False)
    print(f"✅ Added publication types to '{csv_path}': "
          f"{(df['type'].str.contains('Book')).sum()} books/book sections, "
          f"{(df['type'].str.contains('Editorial')).sum()} editorials")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract publications from Semantic Scholar API")
    parser.add_argument("--author", type=str, default=AUTHOR_ID, help="Semantic Scholar Author ID(s), comma-separated")
    parser.add_argument("--output", type=str, default="data/nima.csv", help="Output CSV path")
    parser.add_argument("--api-key", type=str, default=None, help="Semantic Scholar API Key")
    parser.add_argument("--add-types", type=str, default=None, help="Only add publication types to this existing CSV")
    args = parser.parse_args()

    if args.add_types:
        add_publication_types(args.add_types, api_key=args.api_key)
    else:
        run_extraction(author_id=args.author, output_path=args.output, api_key=args.api_key)
