"""
utils.py
--------
Shared utility functions: file paths, result formatting, severity helpers.
"""

import os
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Path helpers — centralise all model/data paths here so nothing is hardcoded
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = MODELS_DIR / "rf_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METRICS_PATH = MODELS_DIR / "model_metrics.json"
DATASET_PATH = DATA_DIR / "dataset.csv"


def models_exist() -> bool:
    """Check if trained model files are present."""
    return MODEL_PATH.exists() and VECTORIZER_PATH.exists()


def load_metrics() -> dict:
    """Load saved model metrics if available."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Severity helpers
# ---------------------------------------------------------------------------

SEVERITY_COLORS = {
    "HIGH": "#FF4B4B",
    "MEDIUM": "#FFA500",
    "LOW": "#FFD700",
    "MINIMAL": "#00CC66",
}

SEVERITY_EMOJI = {
    "HIGH": "🔴",
    "MEDIUM": "🟠",
    "LOW": "🟡",
    "MINIMAL": "🟢",
}


def get_verdict_color(label: str) -> str:
    """Return display color for PHISHING / LEGITIMATE verdict."""
    return "#FF4B4B" if label == "PHISHING" else "#00CC66"


def score_to_severity(score: float) -> str:
    """Convert a 0–100 score to a severity string."""
    if score >= 70:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    elif score >= 15:
        return "LOW"
    return "MINIMAL"


# ---------------------------------------------------------------------------
# Sample examples for the demo button
# ---------------------------------------------------------------------------

SAMPLE_EXAMPLES = [
    {
        "name": "Phishing Email — Fake Bank Alert",
        "text": (
            "URGENT: Your account has been suspended due to suspicious activity! "
            "You must verify your identity immediately or your account will be permanently blocked. "
            "Click the link below to confirm your bank account details and reset your password within 24 hours."
        ),
        "url": "http://secure-banklogin.verify-account.com/confirm?user=reset",
    },
    {
        "name": "Phishing — Prize Scam",
        "text": (
            "Congratulations! You have been selected as our lucky winner for a FREE iPhone 15 Pro! "
            "Claim your prize NOW before it expires! Limited time offer! Click here to enter your details."
        ),
        "url": "http://bit.ly/freeprize-claim2024",
    },
    {
        "name": "Legitimate — Newsletter",
        "text": (
            "Hi there, thank you for subscribing to our monthly newsletter. "
            "Here are the top articles from this week covering technology, science, and culture. "
            "You can unsubscribe at any time using the link at the bottom of this email."
        ),
        "url": "https://newsletter.techdigest.com/issues/weekly-roundup-42",
    },
    {
        "name": "Phishing — Credential Theft",
        "text": (
            "Your Microsoft account shows unauthorized access from an unknown device. "
            "Please verify your credentials immediately to secure your account. "
            "Failure to verify within 48 hours will result in account suspension."
        ),
        "url": "http://microsoft-account-verify.suspicious-domain.net/login.php",
    },
]
