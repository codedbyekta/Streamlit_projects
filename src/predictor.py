"""
predictor.py
------------
Loads the trained ML model and fuses ML probability with heuristic score
to produce a final phishing verdict.

Fusion logic (explainable in interviews):
  - ML gives P(phishing) from 0.0 to 1.0 based on text content
  - Heuristics give a risk score from 0 to 100 based on rules
  - We normalise both to 0-100 and compute a weighted average
  - URL-heavy cases get more heuristic weight; text-only gets more ML weight
  - Final threshold: >= 50 → PHISHING, < 50 → LEGITIMATE

This is simple, deterministic, and easy to explain. No magic.
"""

import pickle
import numpy as np
from pathlib import Path

from src.preprocessing import preprocess
from src.heuristic_engine import run_heuristics
from src.utils import MODEL_PATH, VECTORIZER_PATH, models_exist, score_to_severity


def load_model():
    """Load trained RF model and TF-IDF vectorizer from disk."""
    if not models_exist():
        raise FileNotFoundError(
            "Model files not found. Run `python train_model.py` first."
        )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


# Cache model in memory after first load (avoids re-loading on every Streamlit rerun)
_model_cache = {}


def get_model():
    """Return cached model, loading from disk only once."""
    if "model" not in _model_cache:
        model, vectorizer = load_model()
        _model_cache["model"] = model
        _model_cache["vectorizer"] = vectorizer
    return _model_cache["model"], _model_cache["vectorizer"]


def ml_predict(text: str) -> dict:
    """
    Run the ML pipeline on input text.
    Returns phishing probability and confidence.
    """
    if not text or not text.strip():
        return {"phishing_prob": 0.0, "legitimate_prob": 1.0, "ml_confidence": 0.0, "model_available": False}

    try:
        model, vectorizer = get_model()
        processed = preprocess(text)

        if not processed.strip():
            return {"phishing_prob": 0.0, "legitimate_prob": 1.0, "ml_confidence": 0.0, "model_available": False}

        # Transform text to TF-IDF features and predict
        features = vectorizer.transform([processed])
        proba = model.predict_proba(features)[0]

        # The model is trained with class labels; find which index is "phishing"
        classes = list(model.classes_)
        if 1 in classes:
            phishing_idx = classes.index(1)
        else:
            phishing_idx = 0  # fallback

        phishing_prob = float(proba[phishing_idx])
        legitimate_prob = 1.0 - phishing_prob

        return {
            "phishing_prob": phishing_prob,
            "legitimate_prob": legitimate_prob,
            "ml_confidence": max(phishing_prob, legitimate_prob),
            "model_available": True,
        }

    except FileNotFoundError:
        return {"phishing_prob": 0.0, "legitimate_prob": 1.0, "ml_confidence": 0.0, "model_available": False}


def fuse_scores(ml_phishing_prob: float, heuristic_score: float, has_url: bool, has_text: bool) -> dict:
    """
    Combine ML probability and heuristic score into a final verdict.

    Weighting rationale:
    - URL present: URL heuristics are very reliable for structural anomalies
      → give heuristics more weight (60% heuristic, 40% ML)
    - Text only: ML is better at understanding language patterns
      → give ML more weight (65% ML, 35% heuristic)
    - Only one source available: use that source entirely

    Final score is 0-100. Threshold >= 50 = PHISHING.
    """
    ml_score = ml_phishing_prob * 100  # normalise to 0-100

    if has_text and has_url:
        # Both available: URL analysis adds structural insight
        final_score = 0.4 * ml_score + 0.6 * heuristic_score
    elif has_url and not has_text:
        # URL-only: heuristics are all we have
        final_score = heuristic_score
    elif has_text and not has_url:
        # Text-only: rely more on ML
        final_score = 0.65 * ml_score + 0.35 * heuristic_score
    else:
        final_score = 0.0

    final_label = "PHISHING" if final_score >= 50 else "LEGITIMATE"
    severity = score_to_severity(final_score)

    return {
        "final_score": round(final_score, 1),
        "final_label": final_label,
        "severity": severity,
    }


def predict(text: str = "", url: str = "") -> dict:
    """
    Main prediction function. Takes raw text and/or URL, returns full result dict.

    This is the single entry point called by app.py.
    """
    has_text = bool(text and text.strip())
    has_url = bool(url and url.strip())

    if not has_text and not has_url:
        return {
            "error": "Please provide at least a message or URL to analyze.",
            "final_label": None,
        }

    # --- Layer A: ML prediction ---
    ml_result = ml_predict(text) if has_text else {
        "phishing_prob": 0.0,
        "legitimate_prob": 1.0,
        "ml_confidence": 0.0,
        "model_available": False,
    }

    # --- Layer B: Heuristic engine ---
    heuristic_result = run_heuristics(text=text, url=url)

    # --- Fusion ---
    fusion = fuse_scores(
        ml_phishing_prob=ml_result["phishing_prob"],
        heuristic_score=heuristic_result["combined_score"],
        has_url=has_url,
        has_text=has_text,
    )

    # --- Build short explanation ---
    explanation = _build_explanation(fusion, ml_result, heuristic_result, has_text, has_url)

    return {
        # Core verdict
        "final_label": fusion["final_label"],
        "final_score": fusion["final_score"],
        "severity": fusion["severity"],

        # ML layer
        "ml_phishing_prob": ml_result["phishing_prob"],
        "ml_legitimate_prob": ml_result["legitimate_prob"],
        "ml_confidence": ml_result["ml_confidence"],
        "model_available": ml_result["model_available"],

        # Heuristic layer
        "heuristic_score": heuristic_result["combined_score"],
        "text_heuristic_score": heuristic_result["text_score"],
        "url_heuristic_score": heuristic_result["url_score"],
        "heuristic_severity": heuristic_result["severity"],
        "triggered_indicators": heuristic_result["triggered"],
        "text_categories": heuristic_result["text_categories"],
        "url_categories": heuristic_result["url_categories"],

        # Human-readable
        "explanation": explanation,

        # Pass-through inputs (for Gemini layer)
        "input_text": text,
        "input_url": url,
    }


def _build_explanation(fusion: dict, ml_result: dict, heuristic_result: dict, has_text: bool, has_url: bool) -> str:
    """
    Generate a short rule-based explanation string.
    This is the fallback when Gemini is not available.
    """
    label = fusion["final_label"]
    score = fusion["final_score"]
    severity = fusion["severity"]
    n_indicators = len(heuristic_result["triggered"])

    if label == "PHISHING":
        parts = [f"This content shows {severity.lower()} phishing risk (score: {score:.0f}/100)."]

        if ml_result["model_available"] and ml_result["phishing_prob"] > 0.5:
            parts.append(
                f"The ML classifier assigns {ml_result['phishing_prob']:.0%} probability of phishing "
                f"based on language patterns in the message."
            )

        if n_indicators > 0:
            parts.append(
                f"The heuristic engine flagged {n_indicators} indicator(s) "
                f"(heuristic score: {heuristic_result['combined_score']}/100)."
            )

        if heuristic_result["url_score"] > 30:
            parts.append("The URL shows structural anomalies commonly associated with phishing sites.")

        parts.append(
            "Recommendation: Do not click any links, provide credentials, or download attachments. "
            "Report this message as phishing."
        )
        return " ".join(parts)

    else:
        parts = [f"This content appears legitimate (risk score: {score:.0f}/100)."]

        if n_indicators > 0:
            parts.append(
                f"Note: {n_indicators} minor indicator(s) were detected but below phishing threshold."
            )

        parts.append("Exercise standard caution before clicking links or sharing personal information.")
        return " ".join(parts)
