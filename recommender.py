import pandas as pd
import faiss
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

        # Create unified text field
        self.df["product_text"] = (
            self.df["name"].fillna("") + " " +
            self.df["category"].fillna("") + " " +
            self.df["desc"].fillna("") + " " +
            self.df["color"].fillna("")
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

        # Build embeddings once
        self.embeddings = self.model.encode(
            self.df["product_text"].tolist()
        ).astype("float32")

        # FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

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
        category = str(parsed.get("category", "") or "")

        print(f"[Recommender] Original query: {original_query}")
        print(f"[Recommender] English query used: {english_query}")
        if category:
            print(f"[Recommender] Detected category: {category}")

        # -------------------------
        # STEP 2: CATEGORY FILTER
        # -------------------------
        if category:
            filtered_df = self.df[
                self.df["category"].str.lower().str.contains(category.lower(), na=False)
            ]
        else:
            filtered_df = self.df

        # fallback if empty
        if len(filtered_df) == 0:
            filtered_df = self.df

        # -------------------------
        # STEP 3: EMBEDDINGS FOR FILTERED DATA
        # -------------------------
        texts = filtered_df["product_text"].tolist()
        embeddings = self.model.encode(texts).astype("float32")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)

        # -------------------------
        # STEP 4: SEARCH
        # -------------------------
        query_vec = self.model.encode([query]).astype("float32")

        _, indices = index.search(query_vec, k)

        results = []
        for i in indices[0]:
            results.append(filtered_df.iloc[i].to_dict())

        # -------------------------
        # FINAL OUTPUT
        # -------------------------
        return {
            "original_query": original_query,
            "english_query_used": english_query,
            "groq_parsed": parsed,
            "results": results
        }