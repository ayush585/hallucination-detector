# backend/counterevidence.py
from typing import List, Dict, Any, Optional
import os
from sentence_transformers import SentenceTransformer
import numpy as np

# Optional LLM (OpenAI). If no key, we fall back to simple splitting.
USE_LLM = bool(os.environ.get("OPENAI_API_KEY"))

if USE_LLM:
    from openai import OpenAI
    _client = OpenAI()

def _simple_atomic_claims(answer: str, max_claims: int = 3) -> List[str]:
    # very lightweight fallback: split by '.' and keep short, declarative bits
    parts = [p.strip() for p in answer.split('.') if p.strip()]
    parts = [p for p in parts if 4 < len(p.split()) < 30]
    return parts[:max_claims] or [answer]

def extract_atomic_claims(answer: str, max_claims: int = 3) -> List[str]:
    if not USE_LLM:
        return _simple_atomic_claims(answer, max_claims)

    prompt = (
        "You are a precise information analyst. Break the following answer into "
        f"{max_claims} short, factual, atomically-verifiable claims (no opinions). "
        "Return them as a JSON array of strings ONLY.\n\nAnswer:\n" + answer
    )
    try:
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        txt = resp.choices[0].message.content.strip()
        # very small, safe JSON parse without importing json for robustness
        import json
        claims = json.loads(txt)
        claims = [str(c).strip() for c in claims if str(c).strip()]
        return claims[:max_claims] or _simple_atomic_claims(answer, max_claims)
    except Exception:
        return _simple_atomic_claims(answer, max_claims)

def generate_counter_evidence(
    question: str,
    answer: str,
    retrieval_engine,
    max_claims: int = 3,
    top_k: int = 3,
    contradiction_margin: float = 0.15,
) -> Dict[str, Any]:
    """
    For each atomic claim, retrieve evidence; compute similarity to the claim
    AND similarity to a negated/alternative phrasing; pick passages that
    contradict or fail to support the claim.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    claims = extract_atomic_claims(answer, max_claims=max_claims)

    results: List[Dict[str, Any]] = []
    for claim in claims:
        evi_docs = retrieval_engine.retrieve(claim, top_k=top_k)
        passages = [d["text"] for d in evi_docs]
        if not passages:
            results.append({
                "claim": claim,
                "status": "NoEvidence",
                "support_score": 0.0,
                "contradiction_score": 0.0,
                "evidence": [],
                "note": "No passages retrieved."
            })
            continue

        # Encode
        claim_emb = model.encode([claim], convert_to_numpy=True)
        ev_emb = model.encode(passages, convert_to_numpy=True)

        # Support = similarity(claim, evidence)
        sup = np.dot(ev_emb, claim_emb.T) / (np.linalg.norm(ev_emb, axis=1) * np.linalg.norm(claim_emb))
        support_score = float(np.max(sup))

        # Simple “counter phrasing”: prepend NOT / opposite cue words and compare.
        # (Crude but effective in practice for short claims)
        neg = "It is false that " + claim
        neg_emb = model.encode([neg], convert_to_numpy=True)
        contra = np.dot(ev_emb, neg_emb.T) / (np.linalg.norm(ev_emb, axis=1) * np.linalg.norm(neg_emb))
        contradiction_score = float(np.max(contra))

        # Decide status
        if contradiction_score > (support_score + contradiction_margin):
            status = "LikelyContradicted"
        elif support_score < 0.35:
            status = "Unverifiable"
        else:
            status = "SupportedOrNeutral"

        # Pick top passage indices for each score
        sup_idx = int(np.argmax(sup))
        contra_idx = int(np.argmax(contra))

        results.append({
            "claim": claim,
            "status": status,
            "support_score": round(support_score * 100, 2),
            "contradiction_score": round(contradiction_score * 100, 2),
            "support_passage": passages[sup_idx],
            "counter_passage": passages[contra_idx],
        })

    return {"question": question, "answer": answer, "analysis": results}
