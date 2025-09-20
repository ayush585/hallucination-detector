import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class RetrievalEngine:
    def __init__(self, corpus_path: str):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.corpus = self._load_corpus(corpus_path)  # <— this name exactly
        self.index, self.embeddings = self._build_index()

    def _load_corpus(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_index(self):
        texts = [doc["text"] for doc in self.corpus]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index, embeddings

    def retrieve(self, query: str, top_k: int = 3):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_emb, top_k)
        return [self.corpus[i] for i in indices[0]]
