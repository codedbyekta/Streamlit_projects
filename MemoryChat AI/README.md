# 🧠 AI Multiverse 

A **stateful Streamlit + Gemini chatbot** that remembers conversation history using `st.session_state`.

## ✨ Features

* 💬 Continuous multi-message chat
* 🧠 Persistent conversation history
* 🎭 Multiple AI personalities
* 🎚️ Adjustable personality intensity
* 💻 Native `st.chat_input()`
* 🤖 Gemini API integration

## 🛠️ Tech Stack

* Python
* Streamlit
* Google Gemini API
* python-dotenv

## 🚀 Run

```bash
python -m venv venv
venv\Scripts\activate
pip install streamlit google-genai python-dotenv
streamlit run app.py
```

Create `.env`:

```env
GEMINI_API_KEY=your_api_key_here
```
**Author:** Ekta
