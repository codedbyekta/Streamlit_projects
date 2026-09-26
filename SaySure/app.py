import uuid
import sqlite3
import json
from datetime import datetime

import streamlit as st
import pandas as pd

from prompts import get_question_generator_prompt

from gemini_client import (
    configure_gemini,
    generate_questions,
    transcribe_audio,
    evaluate_answer,
    evaluate_technical_with_followup,
)

# PAGE CONFIGURATION
st.set_page_config(
    page_title="SaySure",
    page_icon="🎙️",
    layout="wide",
)

# CONSTANTS
MAX_FOLLOWUPS_PER_QUESTION = 3

DB_PATH = "viva_panel_history.db"


# DATABASE FOR SAYSURE
def init_saysure_db():
    """
    Creates a separate SaySure table.

    We keep the existing database file so old interview data
    is not destroyed, but SaySure uses its own table.
    """

    conn = sqlite3.connect(DB_PATH)

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

    conn.commit()
    conn.close()


def save_saysure_session(
    practice_type,
    time_limit,
    records,
    avg_clarity,
    avg_conciseness,
    avg_structure,
    avg_understanding,
):
    """
    Save one completed SaySure session.
    """

    conn = sqlite3.connect(DB_PATH)

    conn.execute(
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
            datetime.now().isoformat(
                timespec="seconds"
            ),
        ),
    )

    conn.commit()
    conn.close()


def get_saysure_sessions():

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT
            id,
            practice_type,
            time_limit,
            question_count,
            avg_clarity,
            avg_conciseness,
            avg_structure,
            avg_understanding,
            created_at
        FROM saysure_sessions
        ORDER BY created_at DESC
        """
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def get_saysure_session(session_id):

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    row = conn.execute(
        """
        SELECT *
        FROM saysure_sessions
        WHERE id = ?
        """,
        (session_id,),
    ).fetchone()

    conn.close()

    if row is None:
        return None

    result = dict(row)

    result["records"] = json.loads(
        result["records_json"]
    )

    return result


init_saysure_db()


# SESSION STATE
defaults = {

    # Current application stage
    "stage": "setup",

    # Sidebar navigation
    "view": "practice",

    # Practice settings
    "practice_type": "",
    "time_limit": 60,
    "num_questions": 5,

    # Questions
    "questions": [],
    "current_q_index": 0,

    # Completed question records
    "records": [],

    # Current question state
    #
    # None
    # "followup"
    # "final_feedback"
    #
    "question_state": None,

    # Current answer
    "current_answer": "",

    # Initial communication analysis
    "current_analysis": {},

    # Follow-up conversation
    "followups": [],

    "followups_used": 0,

    # Attempt number
    "attempt_number": 1,

    # Prevent duplicate audio processing
    "processed_audio_hash": None,

    # Gemini
    "api_configured": False,

    # Database
    "saved_to_db": False,

    # Current record save protection
    "current_record_saved": False,
}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# HELPER FUNCTIONS
def create_question(
    text,
    is_followup=False,
):
    """
    Every question gets a unique ID.

    This prevents Streamlit audio widgets from sharing
    state between different questions.
    """

    return {
        "id": str(uuid.uuid4()),
        "text": text,
        "is_followup": is_followup,
    }


def reset_current_question():

    st.session_state.question_state = None

    st.session_state.current_answer = ""

    st.session_state.current_analysis = {}

    st.session_state.followups = []

    st.session_state.followups_used = 0

    st.session_state.processed_audio_hash = None

    st.session_state.current_record_saved = False


def reset_session():

    for key, value in defaults.items():

        st.session_state[key] = value


# PROMPTS USED DIRECTLY BY SAY SURE
def get_communication_analysis_prompt(
    question,
    answer,
):
    """
    Analyze communication separately from understanding.

    Important:
    Poor communication does NOT automatically mean poor understanding.
    """

    return f"""
You are the communication coach for an application called SaySure.

The purpose of SaySure is to help a person practice representing
what they know clearly and concisely under pressure.

You are NOT deciding whether the person would pass an interview.

Question:
{question}

User's spoken answer:
{answer}

Analyze ONLY the communication of this answer.

Evaluate:

1. Clarity
   Is the explanation easy to understand?

2. Conciseness
   Does the user get to the point without unnecessary details?

3. Structure
   Does the answer have a logical flow?

4. Relevance
   Does the answer address the question?

5. Filler words
   Look for words such as:
   um, uh, like, basically, actually, you know, so

6. Rambling
   Does the user repeat themselves or move away from the main point?

IMPORTANT:

Do NOT say that the user does not understand the topic merely because
their communication is poor.

Understanding will be checked separately using follow-up questions.

Return ONLY valid JSON:

{{
    "clarity": <integer 0-10>,
    "conciseness": <integer 0-10>,
    "structure": <integer 0-10>,
    "relevance": <integer 0-10>,
    "rambling": <integer 0-10>,
    "filler_count": <integer>,
    "fillers_detected": [],
    "communication_feedback": "<short specific feedback>",

    "understanding_uncertain": <true or false>,

    "understanding_reason":
        "<why the answer does or does not demonstrate enough understanding>"
}}
"""


def get_followup_prompt(
    question,
    answer,
    analysis,
    previous_followups,
):
    """
    Generate a targeted follow-up to test understanding.

    The follow-up MUST come from something the user actually said.
    """

    return f"""
You are the understanding-verification coach for SaySure.

The user is practicing how to explain what they know.

Original question:
{question}

User's answer:
{answer}

Communication analysis:
{analysis}

Previous follow-ups:
{previous_followups}

Your task is to decide whether another follow-up question is useful
for checking the user's understanding.

Ask a follow-up if:

- The user made a claim without explaining why.
- The user used a technical term without demonstrating its meaning.
- The user gave a memorized-sounding statement.
- The user gave a conclusion without reasoning.
- The answer was too vague to establish understanding.
- The user mentioned an important decision but did not explain it.

Do NOT ask a follow-up just because the user's speaking style was poor.

The follow-up must be based on the user's actual answer.

Good examples:

User:
"I chose XGBoost because it gives better accuracy."

Follow-up:
"Why did XGBoost perform better on your dataset?"

User:
"I used PostgreSQL because it is better."

Follow-up:
"What requirement in your application made PostgreSQL suitable?"

User:
"Random Forest handles nonlinear data."

Follow-up:
"What does nonlinear mean in the context of your traffic data?"

Rules:

- Ask ONE question.
- Keep it natural.
- Do not ask multiple questions at once.
- Do not introduce an unrelated topic.
- Do not judge the user yet.

Return ONLY JSON:

{{
    "follow_up_needed": <true or false>,
    "next_question": "<question or empty string>",
    "reason": "<short reason>"
}}
"""


def get_final_feedback_prompt(
    question,
    answer,
    communication_analysis,
    followups,
):
    """
    Final feedback is generated only after the follow-up phase.
    """

    return f"""
You are the final speaking coach for SaySure.

The user has completed the original answer and the necessary
understanding-checking follow-ups.

Your goal is to help the user improve how they REPRESENT what
they know.

Original question:
{question}

Original answer:
{answer}

Communication analysis:
{communication_analysis}

Follow-up conversation:
{followups}

Evaluate understanding based on the entire conversation.

IMPORTANT:

Do NOT confuse communication problems with lack of understanding.

A user may understand the concept but explain it badly.

A fluent answer also does not automatically prove understanding.

Return ONLY valid JSON:

{{
    "understanding_score": <integer 0-10>,

    "understanding_feedback":
        "<specific explanation of what the user demonstrated>",

    "communication_feedback":
        "<specific feedback about clarity, conciseness and structure>",

    "what_went_well":
        "<specific strength>",

    "what_to_improve":
        "<most important communication improvement>",

    "retry_instruction":
        "<one concrete instruction for the next attempt>"
}}
"""


# SIDEBAR
with st.sidebar:

    st.title("🎙️ SaySure")

    st.caption(
        "Practice expressing what you know "
        "clearly under pressure."
    )

    st.divider()

  
    # API KEY
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Used only for this session and not stored.",
    )

    if api_key:

        if not st.session_state.api_configured:

            try:

                configure_gemini(api_key)

                st.session_state.api_configured = True

                st.success(
                    "Gemini configured ✅"
                )

            except Exception as e:

                st.session_state.api_configured = False

                st.error(
                    f"Gemini configuration failed: {e}"
                )

        else:

            st.success(
                "Gemini configured ✅"
            )

    st.divider()

    st.session_state.view = st.radio(
        "Navigate",
        ["practice", "history"],
        format_func=str.title,
    )

    st.divider()

    if st.button(
        "🔄 Restart Practice"
    ):

        reset_session()

        st.rerun()


# HISTORY
if st.session_state.view == "history":

    st.title("📚 SaySure Practice History")

    sessions = get_saysure_sessions()

    if not sessions:

        st.info(
            "No speaking-practice sessions yet."
        )

    else:

        history_df = pd.DataFrame(
            sessions
        )

        display_df = history_df[
            [
                "id",
                "created_at",
                "practice_type",
                "time_limit",
                "question_count",
                "avg_clarity",
                "avg_conciseness",
                "avg_understanding",
            ]
        ].copy()

        display_df.columns = [
            "ID",
            "Date",
            "Practice Type",
            "Time",
            "Questions",
            "Clarity",
            "Conciseness",
            "Understanding",
        ]

        st.dataframe(
            display_df,
            use_container_width=True,
        )

        st.divider()

        st.subheader(
            "📈 Progress Across Sessions"
        )

        trend_df = history_df[
            [
                "created_at",
                "avg_clarity",
                "avg_conciseness",
                "avg_understanding",
            ]
        ].copy()

        trend_df["created_at"] = pd.to_datetime(
            trend_df["created_at"]
        )

        trend_df = trend_df.set_index(
            "created_at"
        )

        trend_df.columns = [
            "Clarity",
            "Conciseness",
            "Understanding",
        ]

        st.line_chart(
            trend_df
        )

        st.divider()

        selected_id = st.selectbox(
            "View previous session",
            [
                item["id"]
                for item in sessions
            ],
        )

        detail = get_saysure_session(
            selected_id
        )

        if detail:

            st.caption(
                f"{detail['practice_type']} • "
                f"{detail['created_at']}"
            )

            for i, record in enumerate(
                detail["records"]
            ):

                with st.expander(
                    f"Q{i + 1}: {record['question']}"
                ):

                    st.markdown(
                        f"**Your answer:** "
                        f"{record['answer']}"
                    )

                    communication = record.get(
                        "communication",
                        {},
                    )

                    understanding = record.get(
                        "final_feedback",
                        {},
                    )

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Clarity",
                        f"{communication.get('clarity', 0)}/10",
                    )

                    col2.metric(
                        "Conciseness",
                        f"{communication.get('conciseness', 0)}/10",
                    )

                    col3.metric(
                        "Understanding",
                        f"{understanding.get('understanding_score', 0)}/10",
                    )

                    st.markdown(
                        "**Understanding:** "
                        + understanding.get(
                            "understanding_feedback",
                            "",
                        )
                    )

                    st.markdown(
                        "**Communication:** "
                        + understanding.get(
                            "communication_feedback",
                            "",
                        )
                    )

    st.stop()


# STAGE 1 — SETUP
if st.session_state.stage == "setup":

    st.title("🎙️ SaySure")

    st.subheader(
        "Practice how to represent what you know."
    )

    st.write(
        """
        SaySure helps you practice answering questions clearly,
        concisely, and under time pressure.

        If your answer does not demonstrate enough understanding,
        SaySure asks targeted follow-up questions before giving
        the final feedback.
        """
    )

    st.divider()

    with st.form(
        "setup_form"
    ):

        practice_type = st.text_input(
            "What do you want to practice?",
            placeholder=(
                "e.g. Explain my TrafficSense project, "
                "DBMS concepts, DSA, self introduction..."
            )
        )

        time_limit = st.selectbox(
            "Answer time",
            [30, 60, 90],
            index=1,
            format_func=lambda x: f"{x} seconds",
        )

        num_q = st.slider(
            "Number of Questions",
            min_value=1,
            max_value=7,
            value=5,
        )

        submitted = st.form_submit_button(
            "Start Practice →"
        )

    if submitted:

        if not practice_type.strip():
            st.warning("Please tell me what you want to practice.")
            st.stop()

        practice_type = practice_type.strip()

        if not st.session_state.api_configured:

            st.error(
                "⚠️ Add your Gemini API key "
                "in the sidebar first."
            )

        else:

            try:

                with st.spinner(
                    "🤖 Preparing your speaking questions..."
                ):

                    prompt = get_question_generator_prompt(
                        practice_type,
                        num_q,
                    )

                    generated_questions = (
                        generate_questions(prompt)
                    )

                if not generated_questions:

                    st.error(
                        "Gemini did not return any questions."
                    )

                else:

                    st.session_state.practice_type = (
                        practice_type
                    )

                    st.session_state.time_limit = (
                        time_limit
                    )

                    st.session_state.num_questions = (
                        num_q
                    )

                    st.session_state.questions = [
                        create_question(q)
                        for q in generated_questions
                    ]

                    st.session_state.current_q_index = 0

                    st.session_state.records = []

                    st.session_state.attempt_number = 1

                    st.session_state.saved_to_db = False

                    reset_current_question()

                    st.session_state.stage = "practice"

                    st.rerun()

            except Exception as e:

                st.error(
                    "❌ Could not generate questions."
                )

                st.caption(
                    f"Technical error: {e}"
                )


# STAGE 2 — PRACTICE
elif st.session_state.stage == "practice":

    questions = st.session_state.questions

    if not questions:

        st.error(
            "No questions found. Restart practice."
        )

        st.stop()

    idx = st.session_state.current_q_index

    if idx >= len(questions):

        st.session_state.stage = "results"

        st.rerun()

    current_question_data = questions[idx]

    current_question = (
        current_question_data["text"]
    )

    current_question_id = (
        current_question_data["id"]
    )

    # PROGRESS
    st.progress(
        (idx + 1) / len(questions),
        text=(
            f"Question {idx + 1} "
            f"of {len(questions)}"
        ),
    )

    # QUESTION HEADER
    if (
        st.session_state.question_state
        == "followup"
    ):

        followup = (
            st.session_state.followups[-1]
        )

        st.warning(
            "🔎 Follow-up — "
            "this question is checking your understanding."
        )

        display_question = (
            followup["question"]
        )

    else:

        display_question = current_question

        st.caption(
            f"Attempt {st.session_state.attempt_number}"
        )

    st.subheader(
        display_question
    )

    st.info(
        f"⏱️ Try to answer within "
        f"**{st.session_state.time_limit} seconds**."
    )

    st.divider()

    # AUDIO
    audio = st.audio_input(
        "🎙️ Record your answer",
        key=f"audio_{current_question_id}",
    )

    # SUBMIT
    submitted = st.button(
        "Submit Answer →",
        key=f"submit_{current_question_id}",
    )

    if submitted:

        if audio is None:

            st.error(
                "⚠️ Please record your answer first."
            )

        else:

            audio_bytes = audio.getvalue()

            audio_hash = hash(
                audio_bytes
            )

            if (
                audio_hash
                == st.session_state.processed_audio_hash
            ):

                st.warning(
                    "This answer was already processed."
                )

            else:

                st.session_state.processed_audio_hash = (
                    audio_hash
                )

                # STEP 1 — TRANSCRIPTION
                try:

                    with st.spinner(
                        "🎙️ Transcribing your answer..."
                    ):

                        answer_text = transcribe_audio(
                            audio_bytes
                        )

                except Exception as e:

                    st.error(
                        "❌ Audio transcription failed."
                    )

                    st.caption(
                        f"Technical error: {e}"
                    )

                    st.stop()

                if not answer_text.strip():

                    st.error(
                        "Gemini returned an empty transcription."
                    )

                    st.stop()

                # FOLLOW-UP ANSWER
                if (
                    st.session_state.question_state
                    == "followup"
                ):

                    # Save this follow-up answer.
                    current_followup = (
                        st.session_state.followups[-1]
                    )

                    current_followup["answer"] = (
                        answer_text
                    )

                    # Check understanding
                    followup_prompt = (
                        get_communication_analysis_prompt(
                            current_followup["question"],
                            answer_text,
                        )
                    )

                    try:

                        with st.spinner(
                            "🧠 Checking your explanation..."
                        ):

                            followup_analysis = (
                                evaluate_answer(
                                    followup_prompt
                                )
                            )

                    except Exception as e:

                        st.error(
                            "❌ Could not analyze the follow-up."
                        )

                        st.caption(
                            f"Technical error: {e}"
                        )

                        st.stop()

                    current_followup[
                        "analysis"
                    ] = followup_analysis

                    # Ask AI whether understanding is now enough
                    verification_prompt = (
                        get_followup_prompt(
                            current_question,
                            answer_text,
                            followup_analysis,
                            st.session_state.followups,
                        )
                    )

                    try:

                        with st.spinner(
                            "🔎 Verifying your understanding..."
                        ):

                            verification = (
                                evaluate_technical_with_followup(
                                    verification_prompt
                                )
                            )

                    except Exception as e:

                        st.error(
                            "❌ Could not verify understanding."
                        )

                        st.caption(
                            f"Technical error: {e}"
                        )

                        st.stop()

                    # UNDERSTANDING IS SUFFICIENT
                    if not verification.get(
                        "follow_up_needed",
                        False,
                    ):

                        st.session_state.question_state = (
                            "final_feedback"
                        )

                    # ASK ANOTHER FOLLOW-UP
                    elif (
                        st.session_state.followups_used
                        < MAX_FOLLOWUPS_PER_QUESTION
                    ):

                        next_question = (
                            verification.get(
                                "next_question",
                                "",
                            )
                        )

                        if next_question:

                            st.session_state.followups.append(
                                {
                                    "question": next_question,
                                    "answer": "",
                                    "analysis": {},
                                }
                            )

                            st.session_state.followups_used += 1

                            st.session_state.question_state = (
                                "followup"
                            )

                        else:

                            st.session_state.question_state = (
                                "final_feedback"
                            )

                    # MAX FOLLOW-UPS REACHED
                    else:

                        st.session_state.question_state = (
                            "final_feedback"
                        )

                    st.rerun()

                # MAIN ANSWER
                else:

                    st.session_state.current_answer = (
                        answer_text
                    )


                    # Analyze communication

                    communication_prompt = (
                        get_communication_analysis_prompt(
                            current_question,
                            answer_text,
                        )
                    )

                    try:

                        with st.spinner(
                            "🗣️ Analyzing how clearly you communicated..."
                        ):

                            analysis = evaluate_answer(
                                communication_prompt
                            )

                    except Exception as e:

                        st.error(
                            "❌ Could not analyze your answer."
                        )

                        st.caption(
                            f"Technical error: {e}"
                        )

                        st.stop()

                    st.session_state.current_analysis = (
                        analysis
                    )


                    # DECIDE WHETHER UNDERSTANDING NEEDS TESTING

                    if analysis.get(
                        "understanding_uncertain",
                        True,
                    ):

                        followup_prompt = (
                            get_followup_prompt(
                                current_question,
                                answer_text,
                                analysis,
                                [],
                            )
                        )

                        try:

                            with st.spinner(
                                "🔎 Checking whether a follow-up is needed..."
                            ):

                                followup_result = (
                                    evaluate_technical_with_followup(
                                        followup_prompt
                                    )
                                )

                        except Exception as e:

                            st.error(
                                "❌ Could not determine whether "
                                "a follow-up is needed."
                            )

                            st.caption(
                                f"Technical error: {e}"
                            )

                            st.stop()

                        if (
                            followup_result.get(
                                "follow_up_needed",
                                False,
                            )
                            and followup_result.get(
                                "next_question",
                                "",
                            )
                        ):

                            st.session_state.followups = [
                                {
                                    "question": followup_result[
                                        "next_question"
                                    ],
                                    "answer": "",
                                    "analysis": {},
                                }
                            ]

                            st.session_state.followups_used = 1

                            st.session_state.question_state = (
                                "followup"
                            )

                        else:

                            st.session_state.question_state = (
                                "final_feedback"
                            )

                    else:

                        st.session_state.question_state = (
                            "final_feedback"
                        )

                    st.rerun()



# FINAL FEEDBACK
if (
    st.session_state.stage == "practice"
    and st.session_state.question_state
    == "final_feedback"
):

    st.divider()

    st.header(
        "🧠 Your Feedback"
    )

    question = (
        st.session_state.questions[
            st.session_state.current_q_index
        ]["text"]
    )

    answer = (
        st.session_state.current_answer
    )

    communication = (
        st.session_state.current_analysis
    )

    followups = (
        st.session_state.followups
    )

    # FINAL FEEDBACK
    final_prompt = get_final_feedback_prompt(
        question,
        answer,
        communication,
        followups,
    )

    try:

        with st.spinner(
            "🧠 Preparing your final feedback..."
        ):

            final_feedback = evaluate_answer(
                final_prompt
            )

    except Exception as e:

        st.error(
            "❌ Could not generate final feedback."
        )

        st.caption(
            f"Technical error: {e}"
        )

        st.stop()

    # SCORE CARDS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Clarity",
        f"{communication.get('clarity', 0)}/10",
    )

    col2.metric(
        "Conciseness",
        f"{communication.get('conciseness', 0)}/10",
    )

    col3.metric(
        "Structure",
        f"{communication.get('structure', 0)}/10",
    )

    col4.metric(
        "Understanding",
        f"{final_feedback.get('understanding_score', 0)}/10",
    )

    # UNDERSTANDING
    st.subheader(
        "🧠 Did you demonstrate understanding?"
    )

    st.info(
        final_feedback.get(
            "understanding_feedback",
            "No understanding feedback available.",
        )
    )

    # COMMUNICATION
    st.subheader(
        "🗣️ How clearly did you represent it?"
    )

    st.write(
        final_feedback.get(
            "communication_feedback",
            "",
        )
    )

    # WHAT WENT WELL
    st.subheader(
        "✅ What went well"
    )

    st.write(
        final_feedback.get(
            "what_went_well",
            "",
        )
    )

    # WHAT TO IMPROVE
    st.subheader(
        "🔧 What to improve"
    )

    st.write(
        final_feedback.get(
            "what_to_improve",
            "",
        )
    )

    # FILLERS
    filler_count = communication.get(
        "filler_count",
        0,
    )

    fillers = communication.get(
        "fillers_detected",
        [],
    )

    st.markdown(
        f"**Filler words detected:** {filler_count}"
    )

    if fillers:

        st.caption(
            "Detected: "
            + ", ".join(fillers)
        )

    # FOLLOW-UP SUMMARY
    if followups:

        st.subheader(
            "🔎 Understanding Check"
        )

        st.caption(
            f"SaySure asked "
            f"{len(followups)} "
            f"follow-up question(s) "
            f"to verify your understanding."
        )

        for i, followup in enumerate(
            followups
        ):

            with st.expander(
                f"Follow-up {i + 1}"
            ):

                st.markdown(
                    f"**Question:** "
                    f"{followup['question']}"
                )

                st.markdown(
                    f"**Your answer:** "
                    f"{followup.get('answer', '')}"
                )


    # RETRY INSTRUCTION
    st.subheader(
        "🎯 Your next attempt"
    )

    st.success(
        final_feedback.get(
            "retry_instruction",
            "Try the answer again and make it more direct.",
        )
    )


    # SAVE CURRENT RECORD
    if not st.session_state.current_record_saved:

        record = {
            "question": question,

            "answer": answer,

            "communication": communication,

            "followups": followups,

            "final_feedback": final_feedback,

            "attempt": st.session_state.attempt_number,
        }

        st.session_state.records.append(
            record
        )

        st.session_state.current_record_saved = (
            True
        )


    # RETRY / NEXT
    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "🔄 Retry This Question",
            use_container_width=True,
        ):

            st.session_state.attempt_number += 1

            reset_current_question()

            st.rerun()

    with col2:

        if st.button(
            "➡️ Next Question",
            use_container_width=True,
        ):

            next_index = (
                st.session_state.current_q_index
                + 1
            )

            if next_index < len(
                st.session_state.questions
            ):

                st.session_state.current_q_index = (
                    next_index
                )

                st.session_state.attempt_number = 1

                reset_current_question()

                st.rerun()

            else:

                st.session_state.stage = (
                    "results"
                )

                st.rerun()

# RESULTS
if st.session_state.stage == "results":

    st.title(
        "📊 SaySure Practice Summary"
    )

    records = (
        st.session_state.records
    )

    if not records:

        st.warning(
            "No completed answers found."
        )

        st.stop()

    # AVERAGES
    clarity_scores = [
        r["communication"].get(
            "clarity",
            0,
        )
        for r in records
    ]

    conciseness_scores = [
        r["communication"].get(
            "conciseness",
            0,
        )
        for r in records
    ]

    structure_scores = [
        r["communication"].get(
            "structure",
            0,
        )
        for r in records
    ]

    understanding_scores = [
        r["final_feedback"].get(
            "understanding_score",
            0,
        )
        for r in records
    ]

    avg_clarity = (
        sum(clarity_scores)
        / len(clarity_scores)
    )

    avg_conciseness = (
        sum(conciseness_scores)
        / len(conciseness_scores)
    )

    avg_structure = (
        sum(structure_scores)
        / len(structure_scores)
    )

    avg_understanding = (
        sum(understanding_scores)
        / len(understanding_scores)
    )

    # SCORE CARDS
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Clarity",
        f"{avg_clarity:.1f}/10",
    )

    col2.metric(
        "Conciseness",
        f"{avg_conciseness:.1f}/10",
    )

    col3.metric(
        "Structure",
        f"{avg_structure:.1f}/10",
    )

    col4.metric(
        "Understanding",
        f"{avg_understanding:.1f}/10",
    )

    st.divider()

    # QUESTION PERFORMANCE
    st.subheader(
        "📈 Performance Across Questions"
    )

    rows = []

    for i, record in enumerate(
        records
    ):

        rows.append(
            {
                "Question": f"Q{i + 1}",

                "Clarity": record[
                    "communication"
                ].get(
                    "clarity",
                    0,
                ),

                "Conciseness": record[
                    "communication"
                ].get(
                    "conciseness",
                    0,
                ),

                "Structure": record[
                    "communication"
                ].get(
                    "structure",
                    0,
                ),

                "Understanding": record[
                    "final_feedback"
                ].get(
                    "understanding_score",
                    0,
                ),
            }
        )

    summary_df = pd.DataFrame(
        rows
    )

    st.dataframe(
        summary_df,
        use_container_width=True,
    )

    st.line_chart(
        summary_df.set_index(
            "Question"
        )
    )

    st.divider()

    # MAIN PATTERNS
    st.subheader(
        "🎯 What to work on"
    )

    for record in records:

        improvement = (
            record[
                "final_feedback"
            ].get(
                "what_to_improve",
                "",
            )
        )

        if improvement:

            st.write(
                f"• {improvement}"
            )

    # SAVE SESSION
    if not st.session_state.saved_to_db:

        try:

            save_saysure_session(
                practice_type=(
                    st.session_state.practice_type
                ),

                time_limit=(
                    st.session_state.time_limit
                ),

                records=records,

                avg_clarity=avg_clarity,

                avg_conciseness=avg_conciseness,

                avg_structure=avg_structure,

                avg_understanding=(
                    avg_understanding
                ),
            )

            st.session_state.saved_to_db = (
                True
            )

        except Exception as e:

            st.error(
                "Could not save this practice session."
            )

            st.caption(
                f"Database error: {e}"
            )

    if st.session_state.saved_to_db:

        st.success(
            "✅ Practice session saved."
        )

    # NEW PRACTICE
    st.divider()

    if st.button(
        "🎙️ Start New Practice"
    ):

        reset_session()

        st.rerun()
