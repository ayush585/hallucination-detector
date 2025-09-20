import os
import streamlit as st
import requests
from typing import Dict, Any, List

# ---------- Config ----------
BACKEND_URL_DEFAULT = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Hallucination Detector", page_icon="🧠", layout="wide")

st.title("🧠 Lightweight Hallucination Detector")
st.caption("HOGN02 · Retrieval + Confidence Scoring + Short Rationale (+ Counter-Evidence bonus)")

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("⚙️ Backend")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL_DEFAULT, help="FastAPI base URL")
    ping = st.button("Ping /health")

    st.subheader("🔧 Thresholds")
    th_green = st.slider("Verified ≥", min_value=0, max_value=100, value=70, step=1)
    th_yellow = st.slider("Suspected ≥", min_value=0, max_value=100, value=40, step=1)
    if th_yellow > th_green:
        st.warning("Suspected threshold should be ≤ Verified threshold.")
    include_counter = st.checkbox("Generate counter-evidence (bonus)", value=True)

    if ping:
        try:
            r = requests.get(f"{backend_url}/health", timeout=5)
            r.raise_for_status()
            st.success(f"Health OK: {r.json()}")
        except Exception as e:
            st.error(f"Health check failed: {e}")

# ---------- Inputs ----------
col_q, col_a = st.columns(2)
with col_q:
    question = st.text_area("❓ Question", placeholder="Who founded SpaceX?")
with col_a:
    answer = st.text_area("💬 AI Answer (to verify)", placeholder="Jeff Bezos founded SpaceX in 2002.")

go = st.button("Verify Answer", type="primary", use_container_width=True)

# ---------- Helpers ----------
def verdict_badge(verdict: str, confidence_pct: float) -> str:
    v = verdict.lower()
    if v.startswith("verified"):
        color = "🟢"
    elif "suspected" in v:
        color = "🟠"
    else:
        color = "🔴"
    return f"{color} **{verdict}** · {confidence_pct:.2f}%"

def confidence_band(conf_pct: float) -> str:
    if conf_pct >= th_green:
        return "✅ High"
    if conf_pct >= th_yellow:
        return "⚠️ Medium"
    return "❌ Low"

# ---------- Action ----------
if go:
    if not question.strip() or not answer.strip():
        st.warning("Please provide both **Question** and **AI Answer**.")
    else:
        payload = {
            "question": question.strip(),
            "answer": answer.strip(),
            "include_counter": include_counter,
            "threshold_green": th_green / 100.0,   # backend expects 0..1
            "threshold_yellow": th_yellow / 100.0
        }
        try:
            resp = requests.post(f"{backend_url}/verify", json=payload, timeout=60)
            resp.raise_for_status()
            data: Dict[str, Any] = resp.json()

            verdict = data.get("verdict", "Unverifiable")
            conf = float(data.get("confidence", 0.0))  # percent 0..100 from backend
            rationale = data.get("rationale", "")
            evidence = data.get("evidence", [])

            st.subheader("Result")
            st.markdown(verdict_badge(verdict, conf))
            st.write(f"**Confidence**: {conf:.2f}% · {confidence_band(conf)}")
            st.write(f"**Rationale**: {rationale}")

            with st.expander("🔎 Evidence Used (Top Passages)", expanded=False):
                if not evidence:
                    st.info("No evidence returned.")
                else:
                    for i, e in enumerate(evidence, 1):
                        text = e["text"] if isinstance(e, dict) and "text" in e else str(e)
                        st.markdown(f"**{i}.** {text}")

            # --- Counter-evidence block (bonus) ---
            ce = data.get("counter_evidence")
            if ce:
                st.subheader("🛡️ Counter-Evidence Analysis")
                rows: List[Dict[str, Any]] = []
                for item in ce.get("analysis", []):
                    rows.append({
                        "Claim": item.get("claim", ""),
                        "Status": item.get("status", ""),
                        "Support %": item.get("support_score", 0.0),
                        "Contradiction %": item.get("contradiction_score", 0.0),
                        "Support Passage": item.get("support_passage", ""),
                        "Counter Passage": item.get("counter_passage", ""),
                    })
                st.dataframe(rows, use_container_width=True)

            # --- Session history ---
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.append({
                "question": question.strip(),
                "answer": answer.strip(),
                "verdict": verdict,
                "confidence": conf,
                "rationale": rationale
            })

        except Exception as e:
            st.error(f"Verification failed: {e}")

# ---------- History ----------
st.divider()
st.subheader("📈 Session Runs")
if "history" in st.session_state and st.session_state.history:
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.markdown(f"**#{i}** — {verdict_badge(item['verdict'], item['confidence'])}")
        with st.expander("Details"):
            st.write(f"**Q:** {item['question']}")
            st.write(f"**A:** {item['answer']}")
            st.write(f"**Rationale:** {item['rationale']}")
else:
    st.info("Run a verification to see history here.")
