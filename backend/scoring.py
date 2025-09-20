# backend/scoring.py
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from typing import List, Dict

# Lightweight stopword list for coverage calc (kept local to avoid extra deps)
_STOPWORDS = set((
    "a an the of in on at is are was were be been being to for and or if then else with by as from that "
    "this these those it its into over under not no but so do does did done have has had you your we our "
    "they their them he she his her who whom which what when where why how".split()
))

def _keywords(text: str) -> List[str]:
    # alphanum tokens; keep hyphenated words; lowercase; filter stopwords & short tokens
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text.lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 2]

def _safe_cosine(mat_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    """cosine similarity between each row in mat_a and vec_b (1 x d)."""
    denom = (np.linalg.norm(mat_a, axis=1) * np.linalg.norm(vec_b))
    # avoid divide-by-zero
    denom = np.where(denom == 0, 1e-8, denom)
    return np.dot(mat_a, vec_b.T).reshape(-1) / denom

class HallucinationScorer:
    """
    Computes:
      - confidence (semantic similarity between answer and retrieved evidence)
      - coverage% (how many informative answer-terms appear in evidence)
      - rationale text
    Returns evidence texts for transparency.
    """
    def __init__(self, retrieval_engine):
        self.retrieval_engine = retrieval_engine
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate(self, answer: str, evidence_docs: List[Dict]):
        evidence_texts = [doc["text"] for doc in evidence_docs]
        if not evidence_texts:
            return {
                "verdict": "Unverifiable",
                "confidence": 0.0,
                "rationale": "No supporting evidence retrieved.",
                "evidence": [],
                "coverage": 0.0,
                "missing_keywords": [],
                "matched_keywords": []
            }

        # --- Embedding similarity (confidence) ---
        answer_emb = self.model.encode([answer], convert_to_numpy=True)
        evidence_emb = self.model.encode(evidence_texts, convert_to_numpy=True)
        sims = _safe_cosine(evidence_emb, answer_emb)
        avg_conf = float(np.clip(np.mean(sims), -1.0, 1.0))  # keep within [-1,1]

        # --- Keyword coverage (explainability) ---
        ans_keys = list(set(_keywords(answer)))
        ev_concat = " ".join(evidence_texts).lower()
        matched = [k for k in ans_keys if k in ev_concat]
        missing = [k for k in ans_keys if k not in ev_concat]
        coverage = round((len(matched) / max(1, len(ans_keys))) * 100.0, 2)

        # Default verdict buckets (frontend can override via thresholds)
        if avg_conf >= 0.70:
            verdict = "Verified"
        elif avg_conf >= 0.40:
            verdict = "Hallucination Suspected"
        else:
            verdict = "Unverifiable"

        rationale = self._generate_rationale(avg_conf, coverage, missing)

        return {
            "verdict": verdict,
            "confidence": round(avg_conf * 100.0, 2),  # percent
            "rationale": rationale,
            "evidence": evidence_texts,
            "coverage": coverage,                      # percent
            "missing_keywords": missing[:10],
            "matched_keywords": matched[:20],
        }

    def _generate_rationale(self, sim_score: float, coverage: float, missing_terms: List[str]) -> str:
        if sim_score < 0.40:
            miss = ", ".join(missing_terms[:5]) if missing_terms else "—"
            return f"Low semantic match and low coverage ({coverage}%). Missing key terms: {miss}."
        elif sim_score < 0.70:
            return f"Partial support: moderate similarity with coverage {coverage}%. Some terms are weakly supported."
        else:
            return f"Answer aligns well with retrieved evidence; coverage {coverage}%."
