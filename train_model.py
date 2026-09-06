"""
train_model.py
--------------
One-time training script. Loads dataset, preprocesses, trains TF-IDF + Random Forest,
evaluates, and saves model artifacts to models/.

Run this before starting the Streamlit app:
    python train_model.py

Expected dataset format (data/dataset.csv):
    text,label
    "Congratulations you won...", 1
    "Hi team, the meeting is at...", 0

    label: 1 = phishing/spam, 0 = legitimate
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)

from src.preprocessing import batch_preprocess
from src.utils import MODELS_DIR, DATASET_PATH, MODEL_PATH, VECTORIZER_PATH, METRICS_PATH


def load_dataset(path: Path) -> pd.DataFrame:
    """
    Load CSV dataset. Handles common column name variations.
    Expected columns: text (or message/email) + label (or spam/phishing).
    """
    print(f"Loading dataset from {path}...")

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}\n"
            "Please place your dataset CSV at data/dataset.csv\n"
            "See README.md for dataset download instructions."
        )

    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    print(f"  Loaded {len(df)} rows, columns: {list(df.columns)}")

    # Normalize column names — handle common variations
    df.columns = df.columns.str.lower().str.strip()

    # Find text column
    text_col = None
    for candidate in ["text", "message", "email", "body", "content", "sms"]:
        if candidate in df.columns:
            text_col = candidate
            break

    if text_col is None:
        # Try first column as text
        text_col = df.columns[0]
        print(f"  Warning: Could not find text column, using '{text_col}'")

    # Find label column
    label_col = None
    for candidate in ["label", "spam", "phishing", "class", "category", "target"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        label_col = df.columns[1]
        print(f"  Warning: Could not find label column, using '{label_col}'")

    print(f"  Using text column: '{text_col}', label column: '{label_col}'")

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]

    # Normalize labels to 0/1 integers
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    label_map = {
        "spam": 1, "phishing": 1, "1": 1, "yes": 1, "true": 1, "ham": 0,
        "legitimate": 0, "legit": 0, "0": 0, "no": 0, "false": 0,
    }
    df["label"] = df["label"].map(label_map)

    # Drop rows where label couldn't be mapped or text is empty
    original_len = len(df)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]
    df["label"] = df["label"].astype(int)

    print(f"  After cleaning: {len(df)} rows ({original_len - len(df)} dropped)")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


def train(df: pd.DataFrame):
    """
    Full training pipeline:
    1. Preprocess text
    2. Split train/test (80/20 stratified)
    3. Fit TF-IDF vectorizer on training set
    4. Train Random Forest
    5. Evaluate and save
    """
    print("\nPreprocessing text (this may take a minute)...")
    df["processed"] = batch_preprocess(df["text"].tolist())

    # Drop rows where preprocessing yielded empty string
    df = df[df["processed"].str.strip() != ""]
    print(f"  {len(df)} samples after preprocessing")

   # Convert to standard NumPy arrays to avoid Pandas/PyArrow issues
    X = np.asarray(df["processed"].astype(str).tolist(), dtype=object)
    y = np.asarray(df["label"].astype(np.int64).tolist(), dtype=np.int64)

    print(f"X type: {type(X)}, y type: {type(y)}")
    print(f"X dtype: {X.dtype}, y dtype: {y.dtype}")

    # Stratified split: preserves class ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"  Train: {len(X_train)} | Test: {len(X_test)}") 



    # --- TF-IDF Vectorizer ---
    # Why TF-IDF?
    # - Simple, fast, interpretable (each feature = a word or bigram)
    # - ngram_range=(1,2): captures both single words and two-word phrases like "click here"
    # - max_features=10000: vocabulary cap keeps model lightweight
    # - sublinear_tf=True: dampens effect of very frequent terms
    print("\nFitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10_000,
        sublinear_tf=True,
        min_df=2,  # ignore terms that appear in fewer than 2 documents
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"  Vocabulary size: {len(vectorizer.vocabulary_)}")

    # --- Random Forest ---
    # Why Random Forest?
    # - Ensemble of decision trees = robust to overfitting on noisy text data
    # - Provides feature importances for interpretability
    # - Good performance without heavy hyperparameter tuning
    # - n_estimators=200: enough trees for stable performance
    # - class_weight='balanced': handles class imbalance (spam often ~40% of data)
    print("\nTraining Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,          # let trees grow deep (forest handles overfitting)
        min_samples_split=5,
        class_weight="balanced", # important if dataset is imbalanced
        random_state=42,
        n_jobs=-1,               # use all CPU cores
    )
    model.fit(X_train_tfidf, y_train)
    print("  Training complete.")

    # --- Evaluation ---
    print("\nEvaluating on test set...")
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing'])}")

    # --- Save artifacts ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {MODEL_PATH}")

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {VECTORIZER_PATH}")

    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "n_estimators": model.n_estimators,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    print("\nTraining complete. Run `streamlit run app.py` to start the app.")
    return model, vectorizer, metrics


if __name__ == "__main__":
    df = load_dataset(DATASET_PATH)
    train(df)
