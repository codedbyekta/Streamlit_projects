"""
preprocessing.py
----------------
Shared NLP text preprocessing pipeline used by both training and inference.
Uses NLTK for tokenization, stopword removal, and lemmatization.

Why this approach:
- Lemmatization over stemming: produces real words, easier to explain in interviews
- Stopword removal: reduces noise but we KEEP negations (not, no) since they matter in phishing
- Modular design: same function used in train_model.py and predictor.py = no train/serve skew
"""

import re
import string
import nltk


def download_nltk_resources():
    """Download required NLTK data on first use."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


download_nltk_resources()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Negation words to preserve — removing "not" changes meaning entirely
NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "nothing", "nobody"}

# Words that signal phishing but are common stopwords — keep them
PHISHING_SIGNAL_WORDS = {"free", "win", "won", "prize", "click", "now", "urgent", "verify"}

# Build stopword set: standard stopwords minus words we want to keep
_base_stopwords = set(stopwords.words("english"))
STOPWORDS = _base_stopwords - NEGATION_WORDS - PHISHING_SIGNAL_WORDS

_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Basic text cleaning: lowercase, remove URLs, emails, numbers, extra whitespace.
    URLs and emails are replaced with tokens so the model sees them as signals.
    """
    if not text or not isinstance(text, str):
        return ""

    text = text.lower()

    # Replace URLs with a placeholder token (URL presence itself is a phishing signal)
    text = re.sub(r"http\S+|www\.\S+", " urltoken ", text)

    # Replace email addresses with token
    text = re.sub(r"\S+@\S+", " emailtoken ", text)

    # Remove HTML tags if any sneak in
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove numbers (dates, account numbers — too noisy for this classifier)
    text = re.sub(r"\d+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(text: str) -> list:
    """Tokenize cleaned text into word tokens."""
    if not text:
        return []
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    """Remove stopwords while preserving negations and phishing signal words."""
    return [t for t in tokens if t not in STOPWORDS]


def lemmatize(tokens: list) -> list:
    """
    Lemmatize tokens to base form.
    Example: 'verifying' -> 'verify', 'accounts' -> 'account'
    Reduces vocabulary size and helps the model generalize.
    """
    return [_lemmatizer.lemmatize(t) for t in tokens]


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline: clean -> tokenize -> remove stopwords -> lemmatize -> rejoin.
    Returns a single cleaned string (what TF-IDF expects as input).

    Using the same function in both training and inference prevents train/serve skew.
    """
    if not text or not isinstance(text, str):
        return ""

    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)

    # Filter very short tokens (single chars are mostly noise after cleaning)
    tokens = [t for t in tokens if len(t) > 1]

    return " ".join(tokens)


def batch_preprocess(texts: list) -> list:
    """Preprocess a list of texts — used during training."""
    return [preprocess(t) for t in texts]
