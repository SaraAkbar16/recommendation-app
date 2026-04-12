from fastapi import FastAPI
from urllib.parse import parse_qs, unquote, urlparse
from recommender import Recommender
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your wordpress domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

recommender = Recommender("final_dataset.csv")


def _normalize_query(q: str) -> str:
    raw = (q or "").strip()
    if not raw:
        return raw

    decoded = unquote(raw)

    # If the user accidentally sends the entire endpoint in q,
    # extract only the nested q value.
    for candidate in (decoded, raw):
        if "/recommend?" in candidate or "recommend?q=" in candidate:
            target = candidate if "://" in candidate else f"http://local{candidate}"
            parsed = urlparse(target)
            nested_q = parse_qs(parsed.query).get("q", [""])[0]
            nested_q = unquote(nested_q).strip()
            if nested_q:
                return nested_q

    if decoded.lower().startswith("q="):
        return decoded[2:].strip()

    return decoded

@app.get("/")
def home():
    return {"message": "API running"}

@app.get("/recommend")
def recommend(q: str):
    clean_q = _normalize_query(q)
    return recommender.recommend(clean_q)