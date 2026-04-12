import pandas as pd
import numpy as np
import os
import json
import re
from groq import Groq
from sentence_transformers import SentenceTransformer


class Recommender:
    def __init__(self, csv_path):

        # =========================
        # 1. LOAD DATASET
        # =========================
        self.df = pd.read_csv(csv_path)

        # Create unified text field and force robust text-only inputs.
        self.df["product_text"] = (
            self.df["name"].fillna("").astype(str) + " " +
            self.df["category"].fillna("").astype(str) + " " +
            self.df["desc"].fillna("").astype(str) + " " +
            self.df["color"].fillna("").astype(str)
        )
        self.df["product_text"] = self.df["product_text"].map(
            lambda x: x if isinstance(x, str) else str(x)
        )
        # =========================
        # 2. LOAD EMBEDDING MODEL
        # =========================
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # =========================
        # 3. GROQ SETUP
        # =========================
        self.groq_models = list(dict.fromkeys([
            os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ]))
        api_key = os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=api_key) if api_key else None

        # Build embeddings once and reuse for all requests.
        texts = self.df["product_text"].tolist()
        self.embeddings = self.model.encode(texts).astype("float32")

    @staticmethod
    def _topk_l2_indices(query_vec, candidate_embeddings, k):
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

        if not self.client:
            print("[Groq] GROQ_API_KEY not found in current server environment. Using raw query.")
            return {"query": query, "category": ""}

        for model_name in self.groq_models:
            try:
                print(f"[Groq] Trying model: {model_name}")
                response = self.client.chat.completions.create(
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
        query_vec = self.model.encode([query]).astype("float32")[0]
        candidate_embeddings = self.embeddings[candidate_indices]
        local_topk = self._topk_l2_indices(query_vec, candidate_embeddings, k)
        global_topk = candidate_indices[local_topk]

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