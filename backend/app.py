# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional

from retrieval import RetrievalEngine
from scoring import HallucinationScorer
from counterevidence import generate_counter_evidence

app = FastAPI(title="Lightweight Hallucination Detector")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Engines ----
retrieval_engine: Optional[RetrievalEngine] = None
scorer: Optional[HallucinationScorer] = None

class QARequest(BaseModel):
    question: str
    answer: str
    include_counter: bool = False
    threshold_green: float = 0.70   # 0..1
    threshold_yellow: float = 0.40  # 0..1

@app.on_event("startup")
def init_engines():
    global retrieval_engine, scorer
    corpus_path = (Path(__file__).parent.parent / "data" / "corpus.json").resolve()
    retrieval_engine = RetrievalEngine(corpus_path=str(corpus_path))
    scorer = HallucinationScorer(retrieval_engine)

# ---- Friendly root + health ----
@app.get("/")
def root():
    size = len(retrieval_engine.corpus) if retrieval_engine else 0
    return {
        "name": "Lightweight Hallucination Detector",
        "corpus_size": size,
        "routes": ["/health", "/verify", "/demo"],
    }

@app.get("/health")
def health():
    size = len(retrieval_engine.corpus) if retrieval_engine else 0
    return {"status": "ok", "corpus_size": size, "model": "all-MiniLM-L6-v2"}

# ---- Core verification ----
@app.post("/verify")
def verify_qa(request: QARequest):
    assert retrieval_engine is not None and scorer is not None, "Engines not initialized"

    evidence = retrieval_engine.retrieve(request.question)
    base = scorer.evaluate(request.answer, evidence)  # confidence in %

    conf_pct = float(base.get("confidence", 0.0))
    conf = conf_pct / 100.0
    if conf >= request.threshold_green:
        verdict = "Verified"
    elif conf >= request.threshold_yellow:
        verdict = "Hallucination Suspected"
    else:
        verdict = "Unverifiable"
    base["verdict"] = verdict

    if request.include_counter:
        base["counter_evidence"] = generate_counter_evidence(
            request.question, request.answer, retrieval_engine
        )

    return base

# ---- Demo route ----
@app.get("/demo")
def demo():
    """Quick demo for judges: runs a built-in hallucination check."""
    assert retrieval_engine is not None and scorer is not None, "Engines not initialized"
    question = "Who founded SpaceX?"
    answer = "Jeff Bezos founded SpaceX."
    evidence = retrieval_engine.retrieve(question)
    base = scorer.evaluate(answer, evidence)

    conf_pct = float(base.get("confidence", 0.0))
    verdict = "Verified" if conf_pct/100 >= 0.70 else "Hallucination Suspected" if conf_pct/100 >= 0.40 else "Unverifiable"
    base["verdict"] = verdict
    base["question"] = question
    base["answer"] = answer
    return base
