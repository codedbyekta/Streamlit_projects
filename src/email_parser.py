"""
email_parser.py
---------------
Parses uploaded .txt and .eml files into plain text for analysis.

Design decisions:
- Reusable: returns (text, url, error) tuple so app.py stays clean
- Handles both plain .txt and RFC-2822 .eml format
- Extracts the first http/https URL found in the body (passed to URL analyzer)
- Never raises — always returns an error string instead so the UI can display it
"""

import re
import email
from email import policy
from email.parser import BytesParser
from typing import Tuple, Optional


# Supported file extensions
SUPPORTED_EXTENSIONS = {".txt", ".eml"}

# Regex to extract the first URL from text
_URL_RE = re.compile(r"https?://[^\s\"'>]+", re.IGNORECASE)


def extract_first_url(text: str) -> str:
    """Return the first http/https URL found in text, or empty string."""
    match = _URL_RE.search(text)
    return match.group(0).rstrip(".,;)>") if match else ""


def _extract_text_from_eml(raw_bytes: bytes) -> Tuple[str, str, Optional[str]]:
    """
    Parse a raw .eml file and return (body_text, first_url, error).

    Prefers plain-text part; falls back to stripping HTML tags if only HTML
    part exists. Prepends subject and sender lines so the ML classifier has
    full email context.
    """
    try:
        msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    except Exception as exc:
        return "", "", f"Failed to parse .eml file: {exc}"

    parts: list[str] = []

    # Add metadata that phishing classifiers care about
    subject = msg.get("Subject", "")
    sender = msg.get("From", "")
    if subject:
        parts.append(f"Subject: {subject}")
    if sender:
        parts.append(f"From: {sender}")

    body_text = ""

    # Walk MIME parts to get the body
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if "attachment" in disposition:
                continue
            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body_text = payload.decode(charset, errors="replace")
                    break
        # If no plain-text part, try HTML and strip tags
        if not body_text:
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    payload = part.get_payload(decode=True)
                    if payload:
                        charset = part.get_content_charset() or "utf-8"
                        raw_html = payload.decode(charset, errors="replace")
                        body_text = re.sub(r"<[^>]+>", " ", raw_html)
                        body_text = re.sub(r"\s+", " ", body_text).strip()
                    break
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            body_text = payload.decode(charset, errors="replace")
            if msg.get_content_type() == "text/html":
                body_text = re.sub(r"<[^>]+>", " ", body_text)
                body_text = re.sub(r"\s+", " ", body_text).strip()

    parts.append(body_text)
    full_text = "\n".join(parts).strip()

    if not full_text:
        return "", "", "The .eml file appears to have no readable text content."

    first_url = extract_first_url(full_text)
    return full_text, first_url, None


def _extract_text_from_txt(raw_bytes: bytes) -> Tuple[str, str, Optional[str]]:
    """
    Decode a plain-text file and return (text, first_url, error).
    Tries UTF-8, then latin-1 as fallback.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return "", "", "Could not decode the file. Please ensure it is a UTF-8 or Latin-1 text file."

    text = text.strip()
    if not text:
        return "", "", "The uploaded file appears to be empty."

    first_url = extract_first_url(text)
    return text, first_url, None


def parse_uploaded_file(
    file_bytes: bytes,
    filename: str,
) -> Tuple[str, str, Optional[str]]:
    """
    Parse an uploaded file and return (extracted_text, first_url, error_message).

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes from st.file_uploader (use uploaded_file.read()).
    filename : str
        Original filename (used to determine format).

    Returns
    -------
    text : str
        Extracted plain text ready to pass to predict().
    url : str
        First URL found in the file (may be empty string).
    error : str | None
        Error message if parsing failed, None on success.
    """
    if not filename:
        return "", "", "No filename provided."

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in SUPPORTED_EXTENSIONS:
        return (
            "",
            "",
            f"Unsupported file type '{ext}'. Please upload a .txt or .eml file.",
        )

    if not file_bytes:
        return "", "", "The uploaded file is empty."

    if ext == ".eml":
        return _extract_text_from_eml(file_bytes)
    else:  # .txt
        return _extract_text_from_txt(file_bytes)
