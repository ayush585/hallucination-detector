import json
import requests
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class RetrievalEngine:
    def __init__(self, corpus_path: str, wiki_fallback: bool = True, cache_path: str = None):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.corpus = self._load_corpus(corpus_path)
        self.index, self.embeddings = self._build_index()

        self.wiki_fallback = wiki_fallback
        self.cache_path = Path(cache_path) if cache_path else (Path(corpus_path).parent / "wiki_cache.json")

        if not self.cache_path.exists():
            self._write_cache({})
        self._wiki_cache = self._read_cache()

    # ---------------- Corpus ----------------
    def _load_corpus(self, path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _build_index(self):
        texts = [doc["text"] for doc in self.corpus]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)
        return index, embeddings

    # ---------------- Cache ----------------
    def _read_cache(self):
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _write_cache(self, data):
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---------------- Wikipedia Fallback ----------------
    def _fetch_wikipedia_snippets(self, query: str, max_results: int = 3, timeout: int = 5) -> List[Dict[str, str]]:
        key = query.lower().strip()
        if key in self._wiki_cache:
            return self._wiki_cache[key]

        snippets: List[Dict[str, str]] = []
        try:
            # Search step
            sparams = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json",
                "srlimit": max_results,
            }
            r = requests.get("https://en.wikipedia.org/w/api.php", params=sparams, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            hits = data.get("query", {}).get("search", [])

            titles = [h.get("title") for h in hits if h.get("title")]
            if titles:
                tparams = {
                    "action": "query",
                    "prop": "extracts",
                    "explaintext": 1,
                    "exintro": 1,
                    "titles": "|".join(titles),
                    "format": "json",
                }
                r2 = requests.get("https://en.wikipedia.org/w/api.php", params=tparams, timeout=timeout)
                r2.raise_for_status()
                pages = r2.json().get("query", {}).get("pages", {})
                for _, p in pages.items():
                    extract = p.get("extract", "")
                    title = p.get("title", "")
                    if extract:
                        snippets.append({"id": f"wiki:{title}", "text": f"{title}. {extract}"})
        except Exception:
            snippets = []

        self._wiki_cache[key] = snippets
        self._write_cache(self._wiki_cache)
        return snippets

    # ---------------- Retrieval ----------------
    def retrieve(self, query: str, top_k: int = 3, use_answer_hint: str = "") -> List[Dict[str, Any]]:
        q = (query + " " + use_answer_hint).strip() if use_answer_hint else query
        query_emb = self.model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        distances, indices = self.index.search(query_emb, top_k)
        results = [self.corpus[i] for i in indices[0] if i < len(self.corpus)]

        # If no good results and wiki fallback is allowed
        if (not results or all(len(r.get("text", "").strip()) == 0 for r in results)) and self.wiki_fallback:
            wiki_snips = self._fetch_wikipedia_snippets(query, max_results=top_k)
            if wiki_snips:
                return wiki_snips

        return results
