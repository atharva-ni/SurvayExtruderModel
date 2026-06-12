import os
import time
import csv
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
DEFAULT_AUTHOR_ID = os.getenv("SEMANTIC_SCHOLAR_AUTHOR_ID", "144019071")
BASE_URL = "https://api.semanticscholar.org/graph/v1"


def get_all_author_papers(author_id, api_key=None):
    """
    Fetch all papers for a given Semantic Scholar author ID.
    """
    all_papers = []
    offset = 0
    limit = 1000
    headers = {}
    
    # Use provided key or fall back to env key
    key = api_key or API_KEY
    if key:
        headers["x-api-key"] = key
        print("🔑 Using Semantic Scholar API Key.")
    else:
        print("ℹ️ No API Key provided. Running in rate-limited public mode.")

    while True:
        url = f"{BASE_URL}/author/{author_id}/papers"
        params = {
            "limit": limit,
            "offset": offset,
            "fields": "title,year,venue,abstract,authors,citationCount,references"
        }

        try:
            response = requests.get(url, headers=headers, params=params)
            
            # Avoid hitting rate limits (1 request per second for public APIs)
            time.sleep(1)

            if response.status_code != 200:
                print(f"❌ Error ({response.status_code}): {response.text}")
                break

            data = response.json().get("data", [])
            if not data:
                print("✅ No more papers found, stopping.")
                break

            all_papers.extend(data)
            print(f"Fetched {len(data)} papers (Total so far: {len(all_papers)})")

            if len(data) < limit:
                break

            offset += limit

        except Exception as e:
            print(f"❌ HTTP request exception: {e}")
            break

    return all_papers


def extract_paper_info(papers):
    """
    Extract relevant paper information into a flat structure.
    """
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


def save_to_csv(data, filename):
    """
    Save list of paper dicts to a CSV file.
    """
    if not data:
        print("⚠️ No data to save.")
        return

    # Ensure parent directories exist
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ CSV saved successfully to {filename}")


def run_extraction(author_id=None, output_path=None, api_key=None):
    target_author = author_id or DEFAULT_AUTHOR_ID
    target_output = output_path or os.path.join("data", "prof3.csv")
    
    print(f"📡 Fetching papers for Semantic Scholar Author ID: {target_author}")
    papers = get_all_author_papers(target_author, api_key=api_key)
    
    if papers:
        print(f"📊 Total papers found: {len(papers)}")
        final_data = extract_paper_info(papers)
        save_to_csv(final_data, target_output)
    else:
        print("⚠️ No papers retrieved.")


if __name__ == "__main__":
    run_extraction()
