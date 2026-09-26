"""
Lightweight SQLite layer for SaySure speaking-practice history.

Kept intentionally simple (no ORM) since this is a single-file
Streamlit application.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = "viva_panel_history.db"


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create the SaySure session-history table if it does not exist."""
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saysure_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                practice_type TEXT NOT NULL,
                time_limit INTEGER NOT NULL,
                question_count INTEGER NOT NULL,
                avg_clarity REAL NOT NULL,
                avg_conciseness REAL NOT NULL,
                avg_structure REAL NOT NULL,
                avg_understanding REAL NOT NULL,
                records_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )


def save_session(
    practice_type: str,
    time_limit: int,
    records: list[dict],
) -> int:
    """Save one completed SaySure speaking-practice session."""
    if not records:
        raise ValueError("Cannot save an empty session.")

    def score(record: dict, key: str) -> float:
        analysis = record.get("analysis") or record.get("main_analysis") or {}
        try:
            return float(analysis.get(key, 0))
        except (TypeError, ValueError):
            return 0.0

    def understanding_score(record: dict) -> float:
        final_feedback = record.get("final_feedback") or {}
        value = final_feedback.get("understanding_score")

        if value is None:
            value = record.get("understanding_score")

        if value is None:
            analysis = record.get("analysis") or record.get("main_analysis") or {}
            value = analysis.get("understanding_score", 0)

        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    avg_clarity = sum(score(r, "clarity") for r in records) / len(records)
    avg_conciseness = sum(score(r, "conciseness") for r in records) / len(records)
    avg_structure = sum(score(r, "structure") for r in records) / len(records)
    avg_understanding = (
        sum(understanding_score(r) for r in records) / len(records)
    )

    with get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO saysure_sessions (
                practice_type,
                time_limit,
                question_count,
                avg_clarity,
                avg_conciseness,
                avg_structure,
                avg_understanding,
                records_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                practice_type,
                time_limit,
                len(records),
                avg_clarity,
                avg_conciseness,
                avg_structure,
                avg_understanding,
                json.dumps(records),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return cursor.lastrowid


def get_all_sessions() -> list[dict]:
    """Return all saved SaySure sessions, newest first."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, practice_type, time_limit, question_count,
                   avg_clarity, avg_conciseness, avg_structure,
                   avg_understanding, created_at
            FROM saysure_sessions
            ORDER BY created_at DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]


def get_session_detail(session_id: int) -> dict | None:
    """Return one SaySure session including its complete records."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM saysure_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()

        if row is None:
            return None

        result = dict(row)
        result["records"] = json.loads(result["records_json"])
        return result


# Backward-compatible aliases for older app code.
save_interview = save_session
get_all_interviews = get_all_sessions
get_interview_detail = get_session_detail
