"""
llm_helper.py
-------------
Optional Gemini API layer. Converts technical detection output into
human-readable phishing analysis and mitigation advice.

If GEMINI_API_KEY is not set or the API call fails, falls back to
rule-based guidance — the app works fully without this module.
"""

import os
import json


def _get_api_key() -> str:
    """Read Gemini API key from environment."""
    # python-dotenv is used in app.py to load .env — key is available via os.environ
    return os.environ.get("GEMINI_API_KEY", "")


def _build_prompt(result: dict) -> str:
    """
    Build a concise prompt from detection results.
    Keeping the prompt short reduces latency and token cost.
    """
    label = result.get("final_label", "UNKNOWN")
    score = result.get("final_score", 0)
    severity = result.get("severity", "UNKNOWN")
    indicators = result.get("triggered_indicators", [])
    input_url = result.get("input_url", "")
    input_text = result.get("input_text", "")[:300]  # truncate long messages

    indicator_text = ""
    if indicators:
        top = indicators[:5]  # only send top 5 to keep prompt short
        indicator_text = "\n".join(f"- {i['label']}: {i['indicator']}" for i in top)

    prompt = f"""You are a cybersecurity analyst. A phishing detection system analyzed the following content and produced these results.

DETECTION RESULTS:
- Verdict: {label}
- Risk Score: {score}/100
- Severity: {severity}

INPUT CONTENT:
- Message excerpt: {input_text or 'Not provided'}
- URL: {input_url or 'Not provided'}

TOP TRIGGERED INDICATORS:
{indicator_text or 'None'}

Based on these results, provide a concise analysis in exactly this JSON format (no markdown, no extra text):
{{
  "summary": "2-3 sentence plain-English explanation of what this phishing attempt is trying to do",
  "attack_tactic": "Name of the likely phishing tactic (e.g. Credential Phishing, Lottery Scam, Fake Alert)",
  "mitigation": "3 specific, practical steps the user should take right now",
  "risk_level_explanation": "One sentence explaining why this score was assigned"
}}"""

    return prompt


def get_gemini_analysis(result: dict) -> dict:
    """
    Call Gemini API and return structured analysis.
    Falls back to rule-based guidance if API key missing or call fails.
    """
    api_key = _get_api_key()

    if not api_key:
        return _fallback_analysis(result)

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")  # fast, cheap model

        prompt = _build_prompt(result)
        response = model.generate_content(prompt)
        raw_text = response.text.strip()

        # Strip markdown fences if model returns them despite instructions
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]

        parsed = json.loads(raw_text)

        return {
            "source": "gemini",
            "summary": parsed.get("summary", ""),
            "attack_tactic": parsed.get("attack_tactic", ""),
            "mitigation": parsed.get("mitigation", ""),
            "risk_level_explanation": parsed.get("risk_level_explanation", ""),
            "error": None,
        }

    except ImportError:
        return _fallback_analysis(result, note="google-generativeai package not installed.")
    except json.JSONDecodeError:
        # Gemini didn't return valid JSON — use fallback
        return _fallback_analysis(result, note="Could not parse Gemini response.")
    except Exception as e:
        return _fallback_analysis(result, note=f"Gemini API error: {str(e)[:100]}")


def _fallback_analysis(result: dict, note: str = "") -> dict:
    """
    Rule-based fallback guidance when Gemini is unavailable.
    Covers the most common phishing patterns based on triggered categories.
    """
    label = result.get("final_label", "UNKNOWN")
    score = result.get("final_score", 0)
    text_cats = result.get("text_categories", {})
    url_score = result.get("url_heuristic_score", 0)

    # Determine likely attack tactic from indicator categories
    if text_cats.get("credential", 0) > 10:
        tactic = "Credential Phishing"
        summary = (
            "This message appears to be a credential phishing attempt designed to steal your "
            "username and password. Attackers create urgency to prevent you from thinking critically."
        )
    elif text_cats.get("reward", 0) > 10:
        tactic = "Lottery / Prize Scam"
        summary = (
            "This appears to be a prize or lottery scam. No legitimate company distributes prizes "
            "through unsolicited messages. This is designed to collect your personal information."
        )
    elif text_cats.get("threat", 0) > 10:
        tactic = "Account Threat / Scare Tactic"
        summary = (
            "This message uses account suspension threats to create panic and force hasty action. "
            "Legitimate services do not demand immediate action via unsolicited messages."
        )
    elif url_score > 40:
        tactic = "Malicious URL / Spoofed Site"
        summary = (
            "The URL shows structural anomalies typical of phishing sites — spoofed domains, "
            "excessive subdomains, or obfuscated paths designed to impersonate legitimate websites."
        )
    elif label == "PHISHING":
        tactic = "Social Engineering"
        summary = (
            "This content exhibits multiple phishing signals including urgency tactics and "
            "suspicious language patterns consistent with social engineering attacks."
        )
    else:
        tactic = "No Significant Threat Detected"
        summary = (
            "This content does not show strong phishing signals. "
            "Exercise standard online caution as always."
        )

    if label == "PHISHING":
        mitigation = (
            "1. Do not click any links or download attachments from this message. "
            "2. Report it as phishing/spam to your email provider. "
            "3. If you already clicked a link, change your passwords immediately and enable 2FA on affected accounts."
        )
    else:
        mitigation = (
            "1. Continue to verify the sender's identity before sharing personal information. "
            "2. Check URLs carefully before clicking. "
            "3. When in doubt, contact the organization directly through their official website."
        )

    risk_explanation = (
        f"Risk score of {score}/100 based on {len(result.get('triggered_indicators', []))} "
        f"triggered indicators across text analysis and URL inspection."
    )

    return {
        "source": "fallback",
        "summary": summary,
        "attack_tactic": tactic,
        "mitigation": mitigation,
        "risk_level_explanation": risk_explanation,
        "error": note if note else None,
    }
