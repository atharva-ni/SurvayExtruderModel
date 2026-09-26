"""
Minimal OpenAlex API client shared by the dataset builder and author extraction.
Set OPENALEX_API_KEY (free at https://openalex.org) in .env to avoid the small
keyless daily budget; OPENALEX_MAILTO is optional.
"""

import os
import re
import time
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

BASE_URL = "https://api.openalex.org"
MAILTO = os.getenv("OPENALEX_MAILTO", "").strip()
API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()


class OpenAlexBudgetError(RuntimeError):
    pass


def get(endpoint: str, params: dict, api_key: Optional[str] = None, retries: int = 6) -> dict:
    params = dict(params)
    key = (api_key or API_KEY or "").strip()
    if key:
        params["api_key"] = key
    if MAILTO:
        params["mailto"] = MAILTO

    for attempt in range(retries):
        try:
            r = requests.get(f"{BASE_URL}/{endpoint.lstrip('/')}", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 and "budget" in r.text.lower():
                reset = r.headers.get("X-RateLimit-Reset")
                hours = f" (resets in about {int(reset) // 3600} h)" if reset and reset.isdigit() else ""
                raise OpenAlexBudgetError(
                    "OpenAlex daily budget exhausted" + hours + ". Get a free API key at "
                    "https://openalex.org and set OPENALEX_API_KEY in .env (or pass --api-key)."
                )
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"OpenAlex error {r.status_code}: {r.text[:300]}")
        except requests.RequestException as e:
            print(f"   ⚠️  Network error: {e}")
        wait = 2 ** attempt
        print(f"   ⏳ Retrying in {wait}s...")
        time.sleep(wait)
    raise RuntimeError("OpenAlex request failed after retries")


def get_all(endpoint: str, params: dict, api_key: Optional[str] = None) -> List[dict]:
    """Cursor-page through every result."""
    results, cursor = [], "*"
    while cursor:
        data = get(endpoint, {**params, "per-page": 200, "cursor": cursor}, api_key=api_key)
        page = data.get("results", [])
        results.extend(page)
        cursor = data.get("meta", {}).get("next_cursor")
        if not page:
            break
        time.sleep(0.1)
    return results


def rebuild_abstract(inverted: Optional[Dict[str, List[int]]]) -> str:
    if not inverted:
        return ""
    positions = [(pos, word) for word, idxs in inverted.items() for pos in idxs]
    return " ".join(word for _, word in sorted(positions))


def short_id(openalex_id: str) -> str:
    return re.sub(r"^https?://openalex.org/", "", openalex_id or "")
