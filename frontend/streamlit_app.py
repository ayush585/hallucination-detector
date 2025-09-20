# frontend/streamlit_app.py
import os
import json
import time
import streamlit as st
import requests
import pandas as pd
from typing import Dict, Any, List

# -------------------- Config --------------------
BACKEND_URL_DEFAULT = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
st.set_page_config(page_title="Hallucination Detector", page_icon="🧠", layout="wide")

# -------------------- Styles --------------------
CUSTOM_CSS = """
<style>
/* App-wide tweaks */
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1100px;}
/* Header card */
.header-card {
  background: linear-gradient(135deg, #0f172a 0%, #1f2937 60%, #0f172a 100%);
  color: #e5e7eb; border-radius: 16px; padding: 20px 22px; margin-bottom: 10px;
  border: 1px solid rgba(255,255,255,0.08);
}
.header-title {font-size: 1.5rem; font-weight: 700; margin: 0 0 .35rem 0;}
.header-sub {opacity: .9; margin: 0;}
/* Status badges */
.badge {
  display: inline-block; padding: 6px 10px; border-radius: 12px; font-weight: 600; font-size: .92rem;
  border: 1px solid rgba(0,0,0,.08);
}
.badge-green {background: #dcfce7; color: #065f46; border-color: #bbf7d0;}
.badge-amber {background: #fef9c3; color: #854d0e; border-color: #fde68a;}
.badge-red {background: #fee2e2; color: #7f1d1d; border-color: #fecaca;}
/* Cards */
.card {
  border: 1px solid #e5e7eb; border-radius: 14px; padding: 16px 18px; background: #ffffff;
  box-shadow: 0 1px 1px rgba(16,24,40,.04);
}
.card h4 {margin-top: 0;}
/* Footer */
.footer {
  margin-top: 24px; padding-top: 14px; border-top: 1px dashed #e5e7eb; text-align: center; color: #64748b;
  font-size: .92rem;
}
kbd {background:#f1f5f9;border:1px solid #e2e8f0;border-bottom-width:2px;border-radius:6px;padding:2px 6px}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------- Header --------------------
st.markdown(
    """
    <div class="header-card">
      <div class="header-title">🧠 Hallucination Detector</div>
      <p class="header-sub">Lightweight • Transparent • Explainable — Retrieval ✚ Confidence ✚ Rationale ✚ Counter-Evidence</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------- Sidebar (quick backend switch) --------------------
with st.sidebar:
    st.subheader("⚙️ Backend")
    backend_url = st.text_input("Backend URL", value=BACKEND_URL_DEFAULT, help="FastAPI base URL (e.g., http://127.0.0.1:8000)")
    if st.button("Ping /health", use_container_width=True):
        try:
            r = requests.get(f"{backend_url}/health", timeout=5)
            r.raise_for_status()
            st.success(f"Health OK: {r.json()}")
        except Exception as e:
            st.error(f"Health check failed: {e}")

    st.markdown("---")
    st.caption("Built at IEM HackOsis 2.0 🚀")

# -------------------- Helpers --------------------
def verdict_badge(verdict: str, confidence_pct: float) -> str:
    v = verdict.lower()
    if v.startswith("verified"):
        css = "badge badge-green"
    elif "suspected" in v:
        css = "badge badge-amber"
    else:
        css = "badge badge-red"
    return f'<span class="{css}">{verdict} · {confidence_pct:.2f}%</span>'

def confidence_band(conf_pct: float, th_green: int, th_yellow: int) -> str:
    if conf_pct >= th_green:
        return "✅ High"
    if conf_pct >= th_yellow:
        return "⚠️ Medium"
    return "❌ Low"

def highlight_keywords(text: str, matched: List[str]) -> str:
    """Bold matched keywords (case-insensitive)."""
    shown = text
    # Sort by length to avoid partial overlaps
    for k in sorted(set(matched or []), key=len, reverse=True):
        if not k or len(k) < 2:
            continue
        shown = shown.replace(k, f"**{k}**")
        shown = shown.replace(k.capitalize(), f"**{k.capitalize()}**")
        shown = shown.replace(k.upper(), f"**{k.upper()}**")
    return shown

def add_to_history(item: Dict[str, Any]):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append(item)

def get_history_df() -> pd.DataFrame:
    if "history" in st.session_state and st.session_state.history:
        return pd.DataFrame(st.session_state.history)
    return pd.DataFrame(columns=["question", "answer", "verdict", "confidence", "coverage", "rationale", "ts"])

# -------------------- Tabs --------------------
tab_verify, tab_session, tab_settings, tab_about = st.tabs(["🔍 Verify", "📈 Session", "⚙️ Settings", "ℹ️ About"])

# -------------------- Settings (moved thresholds here) --------------------
with tab_settings:
    st.subheader("Tuning")
    col1, col2 = st.columns(2)
    with col1:
        th_green = st.slider("Verified ≥", min_value=0, max_value=100, value=70, step=1)
    with col2:
        th_yellow = st.slider("Suspected ≥", min_value=0, max_value=100, value=40, step=1)
    if th_yellow > th_green:
        st.warning("Suspected threshold should be ≤ Verified threshold.")
    include_counter = st.checkbox("Generate counter-evidence (bonus)", value=True)
    st.caption("Tip: Increase thresholds to be stricter; decrease to be more lenient.")

    st.markdown("---")
    st.subheader("Utilities")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Clear Session History", use_container_width=True):
            st.session_state.pop("history", None)
            st.success("Cleared.")
    with colB:
        df = get_history_df()
        if not df.empty:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download History CSV", data=csv, file_name="hallucination_session.csv",
                               mime="text/csv", use_container_width=True)
        else:
            st.button("Download History CSV", disabled=True, use_container_width=True)

# -------------------- Verify --------------------
with tab_verify:
    st.subheader("Enter Question & Answer")
    c1, c2 = st.columns(2)
    with c1:
        question = st.text_area("❓ Question", placeholder="Who founded SpaceX?")
    with c2:
        answer = st.text_area("💬 AI Answer (to verify)", placeholder="Jeff Bezos founded SpaceX in 2002.")
    go = st.button("Verify Answer", type="primary", use_container_width=True)

    if go:
        if not question.strip() or not answer.strip():
            st.warning("Please provide both **Question** and **AI Answer**.")
        else:
            payload = {
                "question": question.strip(),
                "answer": answer.strip(),
                "include_counter": include_counter,
                "threshold_green": (st.session_state.get("th_green") or 70) / 100.0,
                "threshold_yellow": (st.session_state.get("th_yellow") or 40) / 100.0
            }
            # Use the sliders from Settings directly (not session_state mirror)
            payload["threshold_green"] = th_green / 100.0
            payload["threshold_yellow"] = th_yellow / 100.0

            try:
                t0 = time.time()
                resp = requests.post(f"{backend_url}/verify", json=payload, timeout=60)
                resp.raise_for_status()
                data: Dict[str, Any] = resp.json()
                latency_ms = (time.time() - t0) * 1000

                verdict = data.get("verdict", "Unverifiable")
                conf = float(data.get("confidence", 0.0))      # percent
                rationale = data.get("rationale", "")
                evidence = data.get("evidence", [])
                cov = data.get("coverage")
                matched = data.get("matched_keywords", []) or []
                missing = data.get("missing_keywords", []) or []

                # ----- Status card -----
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f"**Verdict**: {verdict_badge(verdict, conf)}", unsafe_allow_html=True)
                st.write(f"**Confidence**: {conf:.2f}% · {confidence_band(conf, th_green, th_yellow)}")
                if cov is not None:
                    st.write(f"**Coverage**: {float(cov):.2f}% of key terms matched in evidence.")
                st.write(f"**Rationale**: {rationale}")
                st.caption(f"Latency: {latency_ms:.0f} ms · Backend: {backend_url}")
                st.markdown("</div>", unsafe_allow_html=True)

                # ----- Evidence card -----
                with st.expander("🔎 Evidence (Top Passages)", expanded=True):
                    if not evidence:
                        st.info("No evidence returned.")
                    else:
                        for i, e in enumerate(evidence, 1):
                            text = e["text"] if isinstance(e, dict) and "text" in e else str(e)
                            st.markdown(f"**{i}.** {highlight_keywords(text, matched)}")

                    # keyword chips
                    if matched:
                        st.markdown("✅ **Matched terms:** " + ", ".join(f"`{m}`" for m in matched[:12]))
                    if missing:
                        st.markdown("⚠️ **Missing terms:** " + ", ".join(f"`{m}`" for m in missing[:12]))

                # ----- Counter-evidence -----
                ce = data.get("counter_evidence")
                if ce:
                    st.subheader("🛡️ Counter-Evidence")
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
                    st.dataframe(rows, use_container_width=True, height=240)

                # ----- Mini charts (bar for latest) -----
                st.subheader("📊 Latest — Confidence vs Coverage")
                latest_cov = float(cov) if cov is not None else 0.0
                chart_df = pd.DataFrame({"Metric": ["Confidence", "Coverage"], "Value": [conf, latest_cov]}).set_index("Metric")
                st.bar_chart(chart_df)

                # ----- Add to history -----
                add_to_history({
                    "question": question.strip(),
                    "answer": answer.strip(),
                    "verdict": verdict,
                    "confidence": conf,
                    "coverage": float(cov) if cov is not None else None,
                    "rationale": rationale,
                    "ts": pd.Timestamp.utcnow().isoformat(timespec="seconds")
                })

            except Exception as e:
                st.error(f"Verification failed: {e}")

# -------------------- Session --------------------
with tab_session:
    st.subheader("Run History")
    df = get_history_df()
    if df.empty:
        st.info("Run a verification to see history and analytics here.")
    else:
        # Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown('<div class="card"><h4>Runs</h4>', unsafe_allow_html=True)
            st.metric("Total", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card"><h4>Avg Confidence</h4>', unsafe_allow_html=True)
            st.metric("Confidence %", f"{df['confidence'].mean():.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="card"><h4>Avg Coverage</h4>', unsafe_allow_html=True)
            cov_mean = pd.to_numeric(df["coverage"], errors="coerce").fillna(0.0).mean()
            st.metric("Coverage %", f"{cov_mean:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Trend chart
        st.subheader("📉 Trend — Confidence & Coverage")
        df_plot = df.copy()
        df_plot["coverage"] = pd.to_numeric(df_plot.get("coverage"), errors="coerce").fillna(0.0)
        df_plot["confidence"] = pd.to_numeric(df_plot.get("confidence"), errors="coerce").fillna(0.0)
        df_plot["run"] = range(1, len(df_plot) + 1)
        trend_df = df_plot.set_index("run")[["confidence", "coverage"]]
        trend_df.rename(columns={"confidence": "Confidence %", "coverage": "Coverage %"}, inplace=True)
        st.line_chart(trend_df)

        # Table
        st.markdown("### 📋 Runs Table")
        st.dataframe(df[["ts", "verdict", "confidence", "coverage", "question", "answer"]].iloc[::-1], use_container_width=True, height=360)

# -------------------- About --------------------
with tab_about:
    st.subheader("What makes this different?")
    st.markdown(
        """
        **Our USP: Counter-Evidence Mode.** Most checkers only say *wrong* — we show *why*, with side-by-side supporting and contradicting passages.
        
        **How it works**
        - Retrieval over a trusted corpus (and optional Wikipedia fallback)
        - Semantic similarity → **Confidence %**
        - Keyword coverage → **explainability**
        - Short **rationale**
        - Optional **counter-evidence** stress test
        """
    )
    st.markdown("**Ideal for**: chatbots, education tools, enterprise copilots, compliance—anywhere hallucinations hurt trust.")

# -------------------- Footer --------------------
st.markdown(
    "<div class='footer'>v1 • Built with FastAPI + SentenceTransformers + Streamlit • "
    "Press <kbd>R</kbd> to rerun</div>",
    unsafe_allow_html=True,
)
