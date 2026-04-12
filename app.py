from functools import lru_cache
from fastapi import FastAPI
from urllib.parse import parse_qs, unquote, urlparse
from recommender import Recommender
from traslator import roman_urdu_to_english

app = FastAPI()

DATASET_SOURCE = "https://drive.google.com/file/d/1T-j0zvmUSHKHaWW4t2XdqRCylnEefcGz/view"


@lru_cache(maxsize=1)
def get_recommender() -> Recommender:
    return Recommender(DATASET_SOURCE)


def _normalize_query_input(q: str) -> str:
    value = (q or "").strip()
    decoded = unquote(value)

    # Handle accidental nested input like '/recommend?q=joote hai?'.
    if decoded.lower().startswith("/recommend") or "?q=" in decoded.lower():
        parsed = urlparse(decoded)
        query_dict = parse_qs(parsed.query)
        nested_q = query_dict.get("q", [])
        if nested_q and nested_q[0].strip():
            return nested_q[0].strip()

        marker = decoded.lower().find("q=")
        if marker != -1:
            return decoded[marker + 2 :].strip()

    return decoded


@app.get("/")
def home():
    return {
        "message": "Recommendation API is running",
        "recommend_endpoint": "/recommend?q=your+query",
    }


@app.get("/recommend")
def recommend(q: str):
    clean_query = _normalize_query_input(q)

    # STEP 1: convert Roman Urdu → English
    english_query = roman_urdu_to_english(clean_query)

    # STEP 2: get recommendations
    results = get_recommender().recommend(english_query)

    # STEP 3: return JSON for WordPress
    return {
        "original_query": clean_query,
        "english_query": english_query,
        "results": results
    }