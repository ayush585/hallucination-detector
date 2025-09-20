from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from typing import Optional

from retrieval import RetrievalEngine
from scoring import HallucinationScorer
from counterevidence import generate_counter_evidence

app = FastAPI(title="Lightweight Hallucination Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Engines are initialized on startup to avoid reloader/import issues
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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/verify")
def verify_qa(request: QARequest):
    assert retrieval_engine is not None and scorer is not None, "Engines not initialized"

    # 1) Retrieve supporting evidence for the QUESTION (trusted corpus)
    evidence = retrieval_engine.retrieve(request.question)

    # 2) Base evaluation of the ANSWER vs evidence
    base = scorer.evaluate(request.answer, evidence)  # returns confidence in PERCENT

    # 3) Apply frontend thresholds to compute final verdict
    conf_pct = float(base.get("confidence", 0.0))           # 0..100
    conf = conf_pct / 100.0                                  # 0..1
    if conf >= request.threshold_green:
        verdict = "Verified"
    elif conf >= request.threshold_yellow:
        verdict = "Hallucination Suspected"
    else:
        verdict = "Unverifiable"
    base["verdict"] = verdict

    # 4) Optional: counter-evidence analysis (bonus)
    if request.include_counter:
        base["counter_evidence"] = generate_counter_evidence(
            request.question, request.answer, retrieval_engine
        )

    return base
