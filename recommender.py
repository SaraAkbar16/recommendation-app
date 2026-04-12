import json
import os
import re
import shutil
import tempfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class Recommender:
    def __init__(self, dataset_source):
        self.dataset_source = dataset_source
        self.df = None
        self.vectorizer = None
        self.text_matrix = None
        self._groq_client = None
        self._resolved_dataset_path = None

        self.groq_models = list(dict.fromkeys([
            os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ]))

    def _load_dependencies(self):
        import numpy as np
        import pandas as pd

        if self.df is None:
            self.df = pd.read_csv(self._resolve_dataset_path())

            self.df["product_text"] = (
                self.df["name"].fillna("").astype(str) + " " +
                self.df["category"].fillna("").astype(str) + " " +
                self.df["desc"].fillna("").astype(str) + " " +
                self.df["color"].fillna("").astype(str)
            )
            self.df["product_text"] = self.df["product_text"].map(
                lambda x: x if isinstance(x, str) else str(x)
            )

        if self.vectorizer is None or self.text_matrix is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2),
                max_features=50000,
            )
            self.text_matrix = self.vectorizer.fit_transform(self.df["product_text"].tolist())

        return np, pd

    def _get_groq_client(self):
        if self._groq_client is None:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                self._groq_client = False
            else:
                from groq import Groq

                self._groq_client = Groq(api_key=api_key)

        return None if self._groq_client is False else self._groq_client

    def _resolve_dataset_path(self):
        if self._resolved_dataset_path is not None:
            return self._resolved_dataset_path

        source = (self.dataset_source or "").strip()
        if source.startswith(("http://", "https://")):
            download_url = source
            parsed = urlparse(source)

            if "drive.google.com" in parsed.netloc and "/file/d/" in parsed.path:
                match = re.search(r"/file/d/([^/]+)", parsed.path)
                if match:
                    file_id = match.group(1)
                    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

            request = Request(download_url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request) as response:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
                    shutil.copyfileobj(response, temp_file)
                    self._resolved_dataset_path = temp_file.name
        else:
            self._resolved_dataset_path = source

        return self._resolved_dataset_path

    @staticmethod
    def _topk_l2_indices(query_vec, candidate_embeddings, k):
        import numpy as np

        if candidate_embeddings.shape[0] == 0:
            return np.array([], dtype=np.int64)

        k = min(int(k), int(candidate_embeddings.shape[0]))
        if k <= 0:
            return np.array([], dtype=np.int64)

        # Exact squared L2 distance, same metric used by IndexFlatL2.
        dists = np.sum((candidate_embeddings - query_vec) ** 2, axis=1)

        if k == candidate_embeddings.shape[0]:
            return np.argsort(dists)

        topk_unsorted = np.argpartition(dists, k - 1)[:k]
        return topk_unsorted[np.argsort(dists[topk_unsorted])]

    @staticmethod
    def _json_safe_value(value):
        import numpy as np
        import pandas as pd

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float) and not np.isfinite(value):
            return None

        if pd.isna(value):
            return None

        if isinstance(value, dict):
            return {k: Recommender._json_safe_value(v) for k, v in value.items()}

        if isinstance(value, list):
            return [Recommender._json_safe_value(v) for v in value]

        return value

    # =========================
    # 4. GROQ QUERY PROCESSOR
    # =========================
    @staticmethod
    def _extract_json_object(text):
        if not text:
            return None

        cleaned = text.strip()

        # Handle fenced JSON like ```json ... ```
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        match = re.search(r"\{[\s\S]*\}", cleaned)
        if not match:
            return None

        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None

        return None

    def process_query_with_groq(self, query):
        client = self._get_groq_client()

        prompt = f"""
        You are an AI assistant for a product recommender system.

        Convert Roman Urdu / Hindi / casual query into:
        1. Clean English query
        2. Product category (if possible)

        Return ONLY valid JSON like:
        {{
        "query": "...",
        "category": "..."
        }}

        User query:
        {query}
        """

        if not client:
            print("[Groq] GROQ_API_KEY not found in current server environment. Using raw query.")
            return {"query": query, "category": ""}

        for model_name in self.groq_models:
            try:
                print(f"[Groq] Trying model: {model_name}")
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": "You convert multilingual shopping queries to clean English JSON.",
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                )
                content = response.choices[0].message.content if response.choices else ""
                parsed = self._extract_json_object(content)
                if parsed:
                    print(f"[Groq] Success with model: {model_name}")
                    return {
                        "query": str(parsed.get("query", query)),
                        "category": str(parsed.get("category", "") or ""),
                    }
                print(f"[Groq] Model responded but JSON parse failed: {model_name}")
            except Exception as exc:
                print(
                    f"[Groq] Model failed: {model_name} | "
                    f"{type(exc).__name__}: {str(exc)[:300]}"
                )
                continue

        print("[Groq] All models failed. Falling back to raw query.")
        return {"query": query, "category": ""}

    # =========================
    # 5. MAIN RECOMMEND FUNCTION (UPDATED)
    # =========================
    def recommend(self, query, k=5):
        np, _ = self._load_dependencies()

        # -------------------------
        # STEP 1: Groq processing
        # -------------------------
        original_query = query
        parsed = self.process_query_with_groq(query)

        english_query = str(parsed.get("query", query)).strip()
        query = english_query.lower()
        raw_category = parsed.get("category", "")
        category = "" if raw_category is None else str(raw_category).strip()
        if category.lower() in {"none", "null", "nan"}:
            category = ""

        print(f"[Recommender] Original query: {original_query}")
        print(f"[Recommender] English query used: {english_query}")
        if category:
            print(f"[Recommender] Detected category: {category}")

        # -------------------------
        # STEP 2: CATEGORY FILTER
        # -------------------------
        if category:
            mask = self.df["category"].str.lower().str.contains(category.lower(), na=False)
            candidate_indices = np.flatnonzero(mask.to_numpy())
        else:
            candidate_indices = np.arange(len(self.df), dtype=np.int64)

        # fallback if empty
        if candidate_indices.size == 0:
            candidate_indices = np.arange(len(self.df), dtype=np.int64)

        # -------------------------
        # STEP 3: SEARCH IN PRECOMPUTED EMBEDDINGS
        # -------------------------
        query_vec = self.vectorizer.transform([query])
        candidate_matrix = self.text_matrix[candidate_indices]
        similarities = (candidate_matrix @ query_vec.T).toarray().ravel()

        if similarities.size == 0:
            global_topk = np.array([], dtype=np.int64)
        else:
            k = min(int(k), int(similarities.shape[0]))
            if k <= 0:
                global_topk = np.array([], dtype=np.int64)
            elif k == similarities.shape[0]:
                global_topk = candidate_indices[np.argsort(-similarities)]
            else:
                topk_unsorted = np.argpartition(-similarities, k - 1)[:k]
                global_topk = candidate_indices[topk_unsorted[np.argsort(-similarities[topk_unsorted])]]

        results = []
        for idx in global_topk:
            row_dict = self.df.iloc[int(idx)].to_dict()
            results.append(self._json_safe_value(row_dict))

        # -------------------------
        # FINAL OUTPUT
        # -------------------------
        return {
            "original_query": original_query,
            "english_query_used": english_query,
            "groq_parsed": self._json_safe_value(parsed),
            "results": results
        }