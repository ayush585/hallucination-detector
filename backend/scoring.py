from sentence_transformers import SentenceTransformer
import numpy as np

class HallucinationScorer:
    def __init__(self, retrieval_engine):
        self.retrieval_engine = retrieval_engine
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def evaluate(self, answer: str, evidence_docs: list):
        evidence_texts = [doc["text"] for doc in evidence_docs]
        if not evidence_texts:
            return {
                "verdict": "Unverifiable",
                "confidence": 0,
                "rationale": "No supporting evidence retrieved."
            }

        # Encode answer + evidence
        answer_emb = self.model.encode([answer], convert_to_numpy=True)
        evidence_emb = self.model.encode(evidence_texts, convert_to_numpy=True)

        # Compute cosine similarities
        sims = np.dot(evidence_emb, answer_emb.T) / (
            np.linalg.norm(evidence_emb, axis=1) * np.linalg.norm(answer_emb)
        )
        avg_conf = float(np.mean(sims))

        verdict = "Verified" if avg_conf > 0.7 else "Hallucination Suspected"
        rationale = self._generate_rationale(answer, evidence_texts, avg_conf)

        return {
            "verdict": verdict,
            "confidence": round(avg_conf * 100, 2),
            "rationale": rationale,
            "evidence": evidence_texts
        }

    def _generate_rationale(self, answer: str, evidence_texts: list, score: float):
        if score < 0.4:
            return f"The answer likely contains unsupported claims. Evidence did not match."
        elif score < 0.7:
            return f"Partial support found, but some claims in the answer are weakly supported."
        else:
            return f"Answer strongly aligns with retrieved evidence."
