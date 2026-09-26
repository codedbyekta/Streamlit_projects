"""
Thin wrapper around the Google GenAI SDK used by SaySure.

Responsibilities:
1. Generate speaking-practice questions.
2. Transcribe spoken answers.
3. Analyze communication and understanding.
4. Decide whether a targeted follow-up is needed.
5. Generate final feedback after the understanding-check phase.
6. Optionally analyze a camera frame for visible posture notes.

SaySure is a speaking-practice application, so the Gemini helpers are
kept generic and do not make pass/fail interview decisions.
"""

import json
from typing import Any

from google import genai


# Keep this configurable in one place.
MODEL_NAME = "gemini-3.6-flash"

_client = None


def configure_gemini(api_key: str):
    """Configure the shared Gemini client."""
    global _client
    _client = genai.Client(api_key=api_key)


def _get_client():
    """Return the configured Gemini client."""
    if _client is None:
        raise RuntimeError(
            "Gemini client not configured. "
            "Call configure_gemini(api_key) first."
        )
    return _client


def _clean_json(text: str) -> str:
    """
    Remove markdown code fences if Gemini returns JSON inside them.
    """
    text = (text or "").strip()

    if text.startswith("```"):
        parts = text.split("```")

        if len(parts) >= 2:
            text = parts[1].strip()

            if text.lower().startswith("json"):
                text = text[4:].strip()

    return text.strip()


def _parse_json_object(text: str) -> dict:
    """
    Parse a Gemini response as a JSON object.

    This helper is intentionally defensive because model responses can
    occasionally contain whitespace or code fences despite instructions.
    """
    raw = _clean_json(text)

    try:
        data = json.loads(raw)

        if isinstance(data, dict):
            return data

    except (json.JSONDecodeError, TypeError):
        pass

    return {}


def generate_questions(prompt: str) -> list[str]:
    """
    Generate the user's requested speaking-practice questions.
    """
    client = _get_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    raw = _clean_json(response.text)

    try:
        data = json.loads(raw)

        if isinstance(data, list):
            questions = [
                str(item).strip()
                for item in data
                if str(item).strip()
            ]

            if questions:
                return questions

    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback if the model did not return valid JSON.
    questions = [
        line.strip("-• ").strip()
        for line in raw.splitlines()
        if line.strip()
    ]

    return questions[:7]


def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
) -> str:
    """
    Send recorded audio to Gemini and return the transcription.
    """
    client = _get_client()

    from google.genai import types

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type=mime_type,
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            audio_part,
            (
                "Transcribe this audio accurately. "
                "Return only the spoken words as text. "
                "Do not add explanations, corrections, or commentary."
            ),
        ],
    )

    return response.text.strip()


def analyze_posture(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
) -> str:
    """
    Optional camera-frame enrichment.

    Returns only an observation about visible posture/body-language cues.
    It should not make claims about a person's mental state or personality.
    """
    client = _get_client()

    from google.genai import types

    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=mime_type,
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[
            image_part,
            (
                "Give one short, constructive observation about only the "
                "visible posture or body-language cues in this frame that "
                "could affect spoken presentation. Do not infer mental "
                "state, personality, confidence, or emotions."
            ),
        ],
    )

    return response.text.strip()


def evaluate_answer(prompt: str) -> dict:
    """
    Generic structured evaluation helper.

    Kept for compatibility with existing SaySure code that may use
    evaluate_answer() directly.
    """
    client = _get_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    data = _parse_json_object(response.text)

    if data:
        return data

    raw = _clean_json(response.text)

    return {
        "score": 5,
        "strengths": "The response could not be parsed into structured feedback.",
        "gaps": "The model output was not valid JSON.",
        "feedback": raw[:500],
    }


def analyze_communication(prompt: str) -> dict:
    """
    Analyze clarity, conciseness, structure, relevance, and understanding
    uncertainty for the user's main answer.
    """
    client = _get_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    data = _parse_json_object(response.text)

    if data:
        return data

    raw = _clean_json(response.text)

    return {
        "clarity": 5,
        "conciseness": 5,
        "structure": 5,
        "relevance": 5,
        "filler_words": "Could not parse structured analysis.",
        "rambling": "Could not parse structured analysis.",
        "understanding_uncertain": True,
        "reason": raw[:500],
    }


def get_followup_question(prompt: str) -> dict:
    """
    Decide whether another understanding-check follow-up is needed.

    The model is instructed by prompts.py to base the follow-up on the
    user's actual answer.
    """
    client = _get_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    data = _parse_json_object(response.text)

    if data:
        data.setdefault("follow_up_needed", False)
        data.setdefault("next_question", "")
        data.setdefault("reason", "")
        return data

    return {
        "follow_up_needed": False,
        "next_question": "",
        "reason": "Could not parse the follow-up decision.",
    }


def evaluate_technical_with_followup(prompt: str) -> dict:
    """
    Backward-compatible helper.

    Older SaySure code may call this function. It now uses the same
    structured response expected by the updated prompts.
    """
    client = _get_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    data = _parse_json_object(response.text)

    if not data:
        raw = _clean_json(response.text)

        data = {
            "score": 5,
            "strengths": "The response could not be parsed into structured feedback.",
            "gaps": "The model output was not valid JSON.",
            "feedback": raw[:500],
            "follow_up_needed": False,
            "next_question": "",
        }

    data.setdefault("follow_up_needed", False)
    data.setdefault("next_question", "")

    return data


def get_final_feedback(prompt: str) -> dict:
    """
    Generate final speaking-practice feedback after all required
    understanding checks are complete.
    """
    client = _get_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    data = _parse_json_object(response.text)

    if data:
        return data

    raw = _clean_json(response.text)

    return {
        "understanding_score": 5,
        "understanding_feedback": (
            "The final feedback could not be parsed into structured data."
        ),
        "communication_feedback": raw[:500],
        "what_went_well": "",
        "what_to_improve": "",
        "retry_instruction": (
            "Retry the answer with a clear structure: point, explanation, example."
        ),
    }
