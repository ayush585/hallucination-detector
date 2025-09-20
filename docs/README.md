# 🧠 Lightweight Hallucination Detector

A project built for **IEM HackOsis 2.0 🚀** under problem statement **HOGN02**.

This system detects and flags **hallucinations** in AI-generated answers in **real time**, by verifying them against a trusted knowledge base (`corpus.json`) and optionally Wikipedia (fallback). It combines **retrieval**, **confidence scoring**, **rationales**, and a **counter-evidence engine** with a professional **Streamlit dashboard** for interactive analysis.

---

## 📂 Project Structure

```
HALLUCINATION-DETECTOR/
│
├── backend/                  # FastAPI backend
│   ├── app.py                 # Main FastAPI app (routes: /, /health, /verify)
│   ├── counterevidence.py     # Counter-evidence generation logic
│   ├── retrieval.py           # Retrieval engine (local corpus + wiki fallback)
│   ├── scoring.py             # Confidence scoring + rationale generation
│   ├── test_backend.py        # Backend test script
│   ├── utils.py               # Utility helpers
│   └── requirements.txt       # Python dependencies for backend
│
├── data/                     # Knowledge base and caches
│   ├── corpus.json            # Main trusted corpus (facts, tech, history, markets)
│   └── wiki_cache.json        # Cached wiki lookups for efficiency
│
├── docs/                     # Documentation
│   ├── PRD.md                 # Product Requirement Document
│   └── README.md              # Repo guide (this file)
│
├── frontend/                 # Streamlit frontend
│   ├── streamlit_app.py       # Streamlit dashboard UI
│   └── components/            # (Optional) Custom components
│
└── .gitignore                # Ignored files & folders
```

---

## ⚙️ Features

* **Retrieval & Verification** — Fetches supporting evidence from trusted corpus or Wikipedia.
* **Confidence Scoring** — Computes semantic similarity and keyword coverage.
* **Short Rationale** — Explains why an answer is marked as verified/suspected/unverifiable.
* **Counter-Evidence Engine** — Provides contradictory evidence (bonus feature).
* **Streamlit Dashboard** — Polished UI with:

  * Confidence & coverage bar charts 📊
  * Trend lines across multiple runs 📈
  * Evidence & counter-evidence panels 🔍
  * Session history tracking 🗂️
* **Health & Root Endpoints** — Quick visibility of backend status, corpus size, and model info.

---

## 🚀 How It Works

1. **User Input**: Question + AI-generated answer.
2. **Backend (FastAPI)**:

   * Retrieves evidence from `corpus.json`.
   * Falls back to Wikipedia if missing.
   * Scores confidence using `SentenceTransformers`.
   * Generates rationale and (optionally) counter-evidence.
3. **Frontend (Streamlit)**:

   * Displays verdict (✅ Verified / ⚠️ Suspected / ❌ Unverifiable).
   * Visualizes confidence vs. coverage.
   * Provides trend tracking across session runs.

---

## 📦 Setup

### Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

### Frontend

```bash
cd frontend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r ../backend/requirements.txt
streamlit run streamlit_app.py
```

---

## 🔑 USP (Unique Selling Point)

Unlike many heavy hallucination detectors, this project is **lightweight, real-time, and explainable**, with a **counter-evidence module** that not only checks answers but also shows *why they are wrong*.

This makes it:

* ⚡ Fast enough for hackathon-scale real-time QA.
* 🔍 Transparent for judges & mentors.
* 🛡️ More robust than simple similarity checkers.

---

## 📌 About

Developed as part of **IEM HackOsis 2.0** under **Problem Statement HOGN02**.

**Team Goal**: Build a system that balances **speed**, **accuracy**, and **explainability** while remaining **scalable and practical**.

---

✨ Built with: **FastAPI · SentenceTransformers · Streamlit · Wikipedia API**
