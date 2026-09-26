# 🎤 SaySure

**SaySure** is an AI-powered speaking-practice tool that helps users improve both **communication** and **understanding**.

Instead of giving a generic score, SaySure analyzes the user's spoken answer and, when needed, asks a **targeted follow-up question** based on the actual response.

### ✨ Features

* 🎙️ Audio-based answer practice
* 🤖 Gemini-powered question generation
* 📝 Speech-to-text transcription
* 💬 Communication analysis
* 🧠 Understanding checks
* 🔄 Adaptive follow-up questions
* 📊 Final personalized feedback
* 💾 Practice history using SQLite
* ⚡ Bounded follow-up flow

### 🔄 How It Works

```text
Choose a topic
     ↓
Gemini generates questions
     ↓
User answers using audio
     ↓
Answer transcription + analysis
     ↓
Communication + Understanding check
     ↓
Targeted follow-up (if needed)
     ↓
Final feedback
     ↓
Session saved to SQLite
```

### 🛠️ Tech Stack

**Python | Streamlit | Gemini API | SQLite | Pandas | google-genai**

### 📁 Project Structure

```text
saysure/
├── app.py
├── prompts.py
├── gemini_client.py
├── db.py
├── requirements.txt
└── README.md
```

### 🚀 Run Locally

```bash
git clone <repository-url>
cd saysure
pip install -r requirements.txt
streamlit run app.py
```

Add your Gemini API key through Streamlit Secrets :

```toml
GEMINI_API_KEY = "your-key-here"
```

### 🎯 Goal

SaySure is designed for **speaking practice, not interview pass/fail prediction**. It helps users understand not only *what* they are saying, but also *how clearly* they communicate it.


