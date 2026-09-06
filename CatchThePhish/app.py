"""
app.py
------
CatchThePhish — Streamlit web interface.
Main entry point. Run with: streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv
import os

# Load .env file for GEMINI_API_KEY (if present)
load_dotenv()

from src.predictor import predict
from src.llm_helper import get_gemini_analysis
from src.email_parser import parse_uploaded_file
from src.visualization import (
    make_gauge_chart,
    make_ml_probability_chart,
    make_indicator_category_chart,
    make_score_comparison_chart,
    make_confusion_matrix_chart,
)
from src.utils import (
    SAMPLE_EXAMPLES, load_metrics, models_exist,
    SEVERITY_EMOJI, SEVERITY_COLORS, get_verdict_color,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="CatchThePhish",
    page_icon="🎣",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — keep it minimal, just clean up spacing and verdict display
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .verdict-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .indicator-item {
        background: #1e1e2e;
        border-left: 4px solid #FF4B4B;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
        font-size: 0.9em;
    }
    .metric-card {
        background: #1e1e2e;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎣 CatchThePhish")
    st.caption("Phishing Detection & Analysis")
    st.divider()

    st.subheader("Model Status")
    if models_exist():
        st.success("✅ Model loaded")
        metrics = load_metrics()
        if metrics:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
            st.metric("F1 Score", f"{metrics.get('f1_score', 0):.1%}")
            st.metric("Train Samples", f"{metrics.get('train_samples', 'N/A'):,}")
    else:
        st.error("❌ Model not found")
        st.info("Run `python train_model.py` to train the model.")

    st.divider()

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    if gemini_key:
        st.success("✅ Gemini API active")
    else:
        st.info("ℹ️ AI Explanation: Disabled\n(Using rule-based analysis)")

    st.divider()
    st.caption(" Developed by Ekta")
    st.caption("Stack: Python · Scikit-learn · NLTK · TF-IDF · Streamlit")


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("🎣 CatchThePhish")
st.subheader(
    "Real-time Phishing Detection & Analysis"
    "Detect phishing emails and suspicious URLs using Machine Learning, heuristic analysis, and explainable AI insights"
)
st.markdown(
    "Paste a suspicious email/message and/or URL below. "
    "The system analyzes it using a trained ML classifier combined with a heuristic rule engine."
)

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------

tab_analyze, tab_about, tab_model = st.tabs(["🔍 Analyze", "📖 How It Works", "📊 Model Info"])


# ============================================================
# TAB 1 — ANALYZER
# ============================================================

with tab_analyze:

    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("Input")

        # Sample example loader
        with st.expander("📋 Load a sample example", expanded=False):
            for i, example in enumerate(SAMPLE_EXAMPLES):
                if st.button(f"Load: {example['name']}", key=f"sample_{i}"):
                    st.session_state["input_text"] = example["text"]
                    st.session_state["input_url"] = example["url"]
                    st.rerun()

        # --- File upload ---
        with st.expander("📂 Upload .txt or .eml file", expanded=False):
            uploaded_file = st.file_uploader(
                "Drag and drop or browse",
                type=["txt", "eml"],
                label_visibility="collapsed",
                key="file_uploader",
            )
            if uploaded_file is not None:
                file_bytes = uploaded_file.read()
                parsed_text, parsed_url, parse_error = parse_uploaded_file(
                    file_bytes, uploaded_file.name
                )
                if parse_error:
                    st.error(f"❌ {parse_error}")
                else:
                    st.success(f"✅ Loaded **{uploaded_file.name}** ({len(file_bytes):,} bytes)")
                    if st.button("Use this file for analysis", key="use_file_btn"):
                        st.session_state["input_text"] = parsed_text
                        st.session_state["input_url"] = parsed_url if parsed_url else ""
                        st.rerun()



        input_text = st.text_area(
            "Suspicious message / email text",
            value=st.session_state.get("input_text", ""),
            height=180,
            placeholder="Paste the suspicious email or message text here...",
        )

        input_url = st.text_input(
            "Suspicious URL (optional)",
            value=st.session_state.get("input_url", ""),
            placeholder="https://..."
        )

        col_btn1, col_btn2 = st.columns(2)

        with col_btn1:
            analyze_clicked = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with col_btn2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["input_text"] = ""
                st.session_state["input_url"] = ""
                if "result" in st.session_state:
                    del st.session_state["result"]
                st.rerun()

        # Run analysis
        if analyze_clicked:
            text = input_text
            url = input_url



            if not text.strip() and not url.strip():
                st.warning("Please enter a message or URL to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    result = predict(
                        text=text,
                        url=url
                    )
                    if "error" not in result:
                        llm_analysis = get_gemini_analysis(result)
                        result["llm_analysis"] = llm_analysis
                    st.session_state["result"] = result


    # ----------------------------------------
    # Results panel
    # ----------------------------------------

    with col_results:
        st.subheader("Results")

        if "result" not in st.session_state:
            st.info("Enter content on the left and click **Analyze** to see results.")

        else:
            result = st.session_state["result"]

            if "error" in result:
                st.error(result["error"])

            else:
                label = result["final_label"]
                severity = result["severity"]
                final_score = result["final_score"]
                verdict_color = get_verdict_color(label)
                sev_emoji = SEVERITY_EMOJI.get(severity, "")

                # --- Verdict banner ---
                verdict_icon = "🚨" if label == "PHISHING" else "✅"
                st.markdown(
                    f"""<div class="verdict-box" style="background-color:{verdict_color}22; border: 2px solid {verdict_color};">
                        <h2 style="color:{verdict_color}; margin:0;">{verdict_icon} {label}</h2>
                        <p style="margin:4px 0; font-size:1.1em;">Risk Score: <strong>{final_score:.0f}/100</strong> &nbsp;|&nbsp; Severity: {sev_emoji} <strong>{severity}</strong></p>
                    </div>""",
                    unsafe_allow_html=True,
                )

                # --- Score metrics row ---
                m1, m2, m3 = st.columns(3)
                ml_pct = f"{result['ml_phishing_prob']:.1%}" if result["model_available"] else "N/A"
                m1.metric("ML Phishing Prob", ml_pct)
                m2.metric("Heuristic Score", f"{result['heuristic_score']}/100")
                m3.metric("Indicators Triggered", len(result["triggered_indicators"]))

                st.divider()

                # --- LLM / fallback analysis ---
                llm = result.get("llm_analysis", {})
                if llm:
                    source_tag = "🤖 Gemini AI" if llm.get("source") == "gemini" else "🔧 Rule-based Analysis"
                    with st.expander(f"📋 Analysis ({source_tag})", expanded=True):
                        st.markdown(f"**Attack Tactic:** {llm.get('attack_tactic', 'Unknown')}")
                        st.markdown(f"**Summary:** {llm.get('summary', '')}")
                        st.markdown(f"**What to do:** {llm.get('mitigation', '')}")
                        if llm.get("error"):
                            st.caption(f"Note: {llm['error']}")

                # --- Triggered indicators ---
                triggered = result["triggered_indicators"]
                if triggered:
                    with st.expander(f"⚠️ Triggered Indicators ({len(triggered)})", expanded=True):
                        for ind in triggered:
                            cat_color = "#FF4B4B" if ind["points"] >= 15 else "#FFA500" if ind["points"] >= 8 else "#FFD700"
                            st.markdown(
                                f'<div class="indicator-item" style="border-left-color:{cat_color};">'
                                f'<strong>{ind["label"]}</strong> (+{ind["points"]}pts)<br>'
                                f'<small>{ind["indicator"]}</small>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                else:
                    st.success("No significant phishing indicators triggered.")

    # ----------------------------------------
    # Visualizations (below the two columns)
    # ----------------------------------------

    if "result" in st.session_state and "error" not in st.session_state["result"]:
        result = st.session_state["result"]
        st.divider()
        st.subheader("📊 Visualizations")

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            # Score comparison bar chart
            ml_score = result["ml_phishing_prob"] * 100 if result["model_available"] else 0
            fig_comparison = make_score_comparison_chart(
                ml_score=ml_score,
                heuristic_score=result["heuristic_score"],
                final_score=result["final_score"],
            )
            st.plotly_chart(fig_comparison, use_container_width=True)

        with viz_col2:
            # Indicator category breakdown
            fig_indicators = make_indicator_category_chart(
                text_categories=result["text_categories"],
                url_categories=result["url_categories"],
            )
            st.plotly_chart(fig_indicators, use_container_width=True)

        if result["model_available"]:
            viz_col3, viz_col4 = st.columns(2)
            with viz_col3:
                fig_ml = make_ml_probability_chart(
                    result["ml_phishing_prob"],
                    result["ml_legitimate_prob"],
                )
                st.plotly_chart(fig_ml, use_container_width=True)

            with viz_col4:
                fig_gauge = make_gauge_chart(
                    score=result["heuristic_score"],
                    title="Heuristic Risk Score",
                )
                st.plotly_chart(fig_gauge, use_container_width=True)


# ============================================================
# TAB 2 — HOW IT WORKS
# ============================================================

with tab_about:
    st.subheader("How CatchThePhish Works")

    st.markdown("""
    CatchThePhish uses a **two-layer detection system** that combines machine learning with
    deterministic rule-based analysis.
    """)

    with st.expander("🔤 Layer 1: NLP Preprocessing Pipeline", expanded=True):
        st.markdown("""
        Before text reaches the ML classifier, it goes through a cleaning pipeline:

        1. **Lowercasing** — normalizes case so "URGENT" and "urgent" are the same token
        2. **URL/Email tokenization** — replaces URLs with `urltoken` and emails with `emailtoken`
        3. **Punctuation & number removal** — reduces noise
        4. **Tokenization** — splits text into word tokens using NLTK's `word_tokenize`
        5. **Stopword removal** — removes common words (the, a, is...) but **keeps** negations
           (not, no, never) and phishing signal words (urgent, verify, free, win)
        6. **Lemmatization** — maps words to base forms (verifying → verify, accounts → account)

        The same pipeline is used during both training and inference to prevent train/serve skew.
        """)

    with st.expander("🤖 Layer 2A: ML Classifier (TF-IDF + Random Forest)"):
        st.markdown("""
        **TF-IDF Vectorizer**
        - Converts preprocessed text into a numerical feature matrix
        - Uses unigrams + bigrams (ngram_range=(1,2)) to capture "click here", "verify account" patterns
        - 10,000 max features to keep the model lightweight
        - sublinear_tf=True to reduce the effect of very frequent terms

        **Random Forest (200 trees)**
        - Ensemble of decision trees trained on TF-IDF features
        - Robust to overfitting on noisy text data
        - Provides `predict_proba()` — the phishing probability used in the final decision
        - `class_weight='balanced'` handles class imbalance in training data

        **Why not a neural network?**
        For a student project, TF-IDF + RF gives excellent results, is fully interpretable,
        trains in seconds on a laptop, and is easy to explain in interviews.
        """)

    with st.expander("📏 Layer 2B: Heuristic Rule Engine"):
        st.markdown("""
        A deterministic rule-based engine that inspects both **text content** and **URL structure**.

        **Text indicators checked:**
        - Urgency/pressure phrases (act now, within 24 hours, expires soon)
        - Credential harvesting language (verify your account, reset your password)
        - Reward/lottery bait (you have won, claim your prize, free gift)
        - Threat language (account suspended, legal action, unauthorized access)
        - Suspicious CTAs (click here, open the attachment)

        **URL structural checks:**
        - IP address used as domain (e.g., `http://192.168.1.1/login`)
        - `@` symbol in URL (browser ignores everything before `@`)
        - Excessive subdomains (more than 4 levels)
        - Hyphens in domain (paypal-secure-login.com)
        - URL shorteners (bit.ly, tinyurl.com)
        - Suspicious keywords in domain or path (login, verify, banking, password)
        - Non-HTTPS URLs
        - Excessive URL length

        Each triggered rule adds a weighted score. Final heuristic score is 0–100.
        """)

    with st.expander("⚖️ Fusion Logic: How the final verdict is decided"):
        st.markdown("""
        The ML probability and heuristic score are combined using a weighted average:

        ```
        If both text and URL provided:
            final_score = 0.40 × (ML prob × 100) + 0.60 × heuristic_score

        If URL only:
            final_score = heuristic_score

        If text only:
            final_score = 0.65 × (ML prob × 100) + 0.35 × heuristic_score
        ```

        **Threshold: final_score ≥ 50 → PHISHING**

        URL heuristics get more weight when a URL is present because structural
        anomalies (IP address, excessive subdomains, @-symbol) are highly reliable
        indicators that the ML classifier cannot directly observe.
        """)

    with st.expander("🤖 Optional Gemini AI Layer"):
        st.markdown("""
        If a `GEMINI_API_KEY` is configured in `.env`, the detection output is sent to
        Gemini 1.5 Flash to generate:
        - Human-readable phishing summary
        - Likely attack tactic identification
        - Specific mitigation steps

        If the key is absent or the API call fails, the system falls back to rule-based
        guidance derived from the triggered indicator categories.

        The app works fully without Gemini — it is an enhancement, not a dependency.
        """)


# ============================================================
# TAB 3 — MODEL INFO
# ============================================================

with tab_model:
    st.subheader("Model Information")

    metrics = load_metrics()

    if not metrics:
        st.warning("No model metrics found. Train the model first with `python train_model.py`.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        col2.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        col3.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        col4.metric("F1 Score", f"{metrics.get('f1_score', 0):.1%}")

        st.divider()

        detail_col1, detail_col2 = st.columns(2)
        with detail_col1:
            fig_cm = make_confusion_matrix_chart(metrics)
            st.plotly_chart(fig_cm, use_container_width=True)

        with detail_col2:
            st.markdown("**Training Details**")
            st.json({
                "Train samples": metrics.get("train_samples"),
                "Test samples": metrics.get("test_samples"),
                "Vocabulary size": metrics.get("vocabulary_size"),
                "RF trees (n_estimators)": metrics.get("n_estimators"),
                "TF-IDF ngram range": "(1, 2)",
                "TF-IDF max features": 10000,
            })

    st.divider()
    st.markdown("""
    **Known Limitations:**
    - Trained on SMS/email data; may not generalize perfectly to all phishing formats
    - URL analysis is structural only — does not fetch or render the actual page
    - Heuristic rules can be evaded by attackers who know the patterns
    - Model performance depends heavily on training data quality and balance
    """)
