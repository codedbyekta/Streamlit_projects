"""
heuristic_engine.py
--------------------
Deterministic rule-based phishing detection engine.
Inspects both text content and URL structure for phishing indicators.

Why heuristics alongside ML?
- ML models miss unseen patterns and new phishing variants
- Rules catch structural URL anomalies the text classifier can't see
- Transparent, debuggable, and easy to explain in interviews
- Combined system is more robust than either alone
"""

import re
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# TEXT INDICATOR RULES
# Each rule is: (pattern_or_checker, score_points, label, category)
# ---------------------------------------------------------------------------

# Urgency and pressure phrases — classic social engineering tactics
URGENCY_PHRASES = [
    "act now", "act immediately", "urgent", "immediately", "right away",
    "limited time", "expires soon", "last chance", "don't delay", "respond now",
    "your account will be", "your account has been", "action required",
    "respond within", "hours to respond", "24 hours", "48 hours",
]

# Credential harvesting intent
CREDENTIAL_PHRASES = [
    "verify your account", "verify your identity", "confirm your details",
    "update your information", "update your account", "confirm your password",
    "enter your password", "reset your password", "login to verify",
    "validate your account", "re-enter your", "provide your",
    "bank account", "social security", "credit card number",
]

# Reward / lottery bait
REWARD_PHRASES = [
    "you have won", "you've won", "congratulations you", "claim your prize",
    "free gift", "you are selected", "lucky winner", "cash prize",
    "lottery winner", "winner of", "you have been chosen",
    "free iphone", "free money", "bonus reward",
]

# Threat language
THREAT_PHRASES = [
    "account suspended", "account terminated", "account blocked",
    "legal action", "will be prosecuted", "sued", "hacked",
    "unauthorized access", "suspicious activity detected",
    "your account is compromised",
]

# Suspicious call-to-action language
SUSPICIOUS_CTA = [
    "click here", "click the link", "click below", "click this link",
    "follow the link", "open the attachment", "download the file",
    "open the link", "tap here", "tap the link",
]


def analyze_text(text: str) -> dict:
    """
    Scan message text for phishing indicators.
    Returns a dict with score, triggered rules, and category breakdown.
    """
    if not text or not isinstance(text, str):
        return {
            "score": 0,
            "triggered": [],
            "categories": {},
        }

    text_lower = text.lower()
    triggered = []
    score = 0

    def check_phrases(phrase_list, label, category, points_each=10):
        hits = [p for p in phrase_list if p in text_lower]
        for hit in hits:
            triggered.append({
                "indicator": hit,
                "category": category,
                "label": label,
                "points": points_each,
            })
        return min(len(hits) * points_each, 30)  # cap per category

    score += check_phrases(URGENCY_PHRASES, "Urgency/Pressure", "urgency", 10)
    score += check_phrases(CREDENTIAL_PHRASES, "Credential Harvesting", "credential", 15)
    score += check_phrases(REWARD_PHRASES, "Reward/Lottery Bait", "reward", 10)
    score += check_phrases(THREAT_PHRASES, "Threat Language", "threat", 12)
    score += check_phrases(SUSPICIOUS_CTA, "Suspicious CTA", "cta", 8)

    # Excessive capitalization (SHOUTING is common in phishing)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    if caps_ratio > 0.3 and len(text) > 20:
        score += 8
        triggered.append({
            "indicator": f"High capitalization ratio ({caps_ratio:.0%})",
            "category": "formatting",
            "label": "Excessive Caps",
            "points": 8,
        })

    # Suspicious number of exclamation marks
    exclaim_count = text.count("!")
    if exclaim_count >= 3:
        score += 5
        triggered.append({
            "indicator": f"{exclaim_count} exclamation marks",
            "category": "formatting",
            "label": "Excessive Punctuation",
            "points": 5,
        })

    # Build category breakdown
    categories = {}
    for item in triggered:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + item["points"]

    return {
        "score": min(score, 100),
        "triggered": triggered,
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# URL INDICATOR RULES
# ---------------------------------------------------------------------------

# Suspicious keywords that appear in phishing domains/paths
SUSPICIOUS_URL_KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update",
    "confirm", "banking", "paypal", "amazon", "apple", "microsoft",
    "google", "facebook", "netflix", "support", "password", "credential",
    "wallet", "recovery", "suspended", "unusual", "alert",
]

# Known URL shortener domains (these hide the real destination)
URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "shorturl.at", "tiny.cc",
]


def analyze_url(url: str) -> dict:
    """
    Inspect URL structure for phishing indicators.
    Returns a dict with score, triggered rules, and category breakdown.
    """
    if not url or not isinstance(url, str):
        return {
            "score": 0,
            "triggered": [],
            "categories": {},
        }

    url = url.strip()

    # Add scheme if missing so urlparse works correctly
    if not url.startswith(("http://", "https://")):
        url = "http://" + url

    triggered = []
    score = 0

    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        full_url = url.lower()
    except Exception:
        return {"score": 30, "triggered": [{"indicator": "Unparseable URL", "category": "structure", "label": "Invalid URL", "points": 30}], "categories": {"structure": 30}}

    def add_flag(indicator, category, label, points):
        triggered.append({
            "indicator": indicator,
            "category": category,
            "label": label,
            "points": points,
        })
        return points

    # --- High-severity structural anomalies ---

    # IP address used instead of domain name
    if re.match(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$", domain):
        score += add_flag("IP address used as domain", "structure", "IP Address URL", 35)

    # @ symbol in URL (tricks browsers — everything before @ is ignored)
    if "@" in url:
        score += add_flag("'@' symbol in URL", "structure", "URL Contains @", 30)

    # --- Domain anomalies ---

    # Excessive subdomains (legit sites rarely have 3+ levels)
    subdomain_parts = domain.split(".")
    if len(subdomain_parts) > 4:
        score += add_flag(
            f"Excessive subdomains ({len(subdomain_parts) - 2} levels): {domain}",
            "domain", "Excessive Subdomains", 20
        )

    # Hyphens in domain (common in spoofing: paypal-secure.com)
    hyphen_count = domain.count("-")
    if hyphen_count >= 2:
        score += add_flag(
            f"{hyphen_count} hyphens in domain",
            "domain", "Hyphen-Heavy Domain", 15
        )

    # Domain is a URL shortener — hides real destination
    if any(shortener in domain for shortener in URL_SHORTENERS):
        score += add_flag(
            f"URL shortener detected: {domain}",
            "domain", "URL Shortener", 20
        )

    # HTTP (not HTTPS) — no encryption
    if parsed.scheme == "http":
        score += add_flag("Non-HTTPS URL (HTTP)", "structure", "No SSL/HTTPS", 10)

    # --- Keyword-based detection in domain + path ---

    # Suspicious keywords in domain
    domain_keyword_hits = [kw for kw in SUSPICIOUS_URL_KEYWORDS if kw in domain]
    if domain_keyword_hits:
        score += add_flag(
            f"Suspicious keywords in domain: {', '.join(domain_keyword_hits[:3])}",
            "keywords", "Suspicious Domain Keywords", min(len(domain_keyword_hits) * 10, 25)
        )

    # Suspicious keywords in path
    path_keyword_hits = [kw for kw in SUSPICIOUS_URL_KEYWORDS if kw in path]
    if path_keyword_hits:
        score += add_flag(
            f"Suspicious keywords in path: {', '.join(path_keyword_hits[:3])}",
            "keywords", "Suspicious Path Keywords", min(len(path_keyword_hits) * 7, 20)
        )

    # --- Length heuristic ---
    if len(url) > 100:
        score += add_flag(
            f"Very long URL ({len(url)} chars)",
            "structure", "Excessive URL Length", 10
        )

    # Excessive dots in full URL (can indicate obfuscation)
    dot_count = full_url.count(".")
    if dot_count > 6:
        score += add_flag(
            f"Unusually many dots in URL ({dot_count})",
            "structure", "Suspicious Dot Pattern", 8
        )

    # Build category breakdown
    categories = {}
    for item in triggered:
        cat = item["category"]
        categories[cat] = categories.get(cat, 0) + item["points"]

    return {
        "score": min(score, 100),
        "triggered": triggered,
        "categories": categories,
    }


def run_heuristics(text: str = "", url: str = "") -> dict:
    """
    Master function: runs both text and URL analysis, combines scores.

    Combination logic:
    - If only text: use text score
    - If only URL: use URL score
    - If both: weighted average (URL gets slightly more weight since
      structural anomalies are more reliable than keyword matches)
    """
    text_result = analyze_text(text) if text else {"score": 0, "triggered": [], "categories": {}}
    url_result = analyze_url(url) if url else {"score": 0, "triggered": [], "categories": {}}

    has_text = bool(text and text.strip())
    has_url = bool(url and url.strip())

    if has_text and has_url:
        combined_score = int(0.45 * text_result["score"] + 0.55 * url_result["score"])
    elif has_url:
        combined_score = url_result["score"]
    else:
        combined_score = text_result["score"]

    # All triggered indicators combined
    all_triggered = text_result["triggered"] + url_result["triggered"]

    # Severity label based on score
    if combined_score >= 70:
        severity = "HIGH"
    elif combined_score >= 40:
        severity = "MEDIUM"
    elif combined_score >= 15:
        severity = "LOW"
    else:
        severity = "MINIMAL"

    return {
        "combined_score": combined_score,
        "text_score": text_result["score"],
        "url_score": url_result["score"],
        "severity": severity,
        "triggered": all_triggered,
        "text_categories": text_result["categories"],
        "url_categories": url_result["categories"],
    }
