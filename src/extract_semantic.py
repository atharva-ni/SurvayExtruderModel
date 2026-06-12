import os
import time
import csv
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "bTfTpZDCqB6NtYTn3WUyc7TyGaGLBLuygGhwXkGd")
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
            "fields": "title,year,venue,abstract,authors,citationCount,references"
        }

        response = requests.get(url, headers=headers, params=params)
        time.sleep(1)  # avoid hitting rate limits

        if response.status_code != 200:
            print("Error:", response.text)
            break

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
            "year": p.get("year")
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
    target_author = author_id or AUTHOR_ID
    target_output = output_path or "data/nima.csv"
    
    papers = get_all_author_papers(target_author, api_key=api_key)
    print(f"Total papers found: {len(papers)}")

    final_data = extract_paper_info(papers)
    save_to_csv(final_data, target_output)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract publications from Semantic Scholar API")
    parser.add_argument("--author", type=str, default=AUTHOR_ID, help="Semantic Scholar Author ID")
    parser.add_argument("--output", type=str, default="data/nima.csv", help="Output CSV path")
    parser.add_argument("--api-key", type=str, default=None, help="Semantic Scholar API Key")
    args = parser.parse_args()

    run_extraction(author_id=args.author, output_path=args.output, api_key=args.api_key)
