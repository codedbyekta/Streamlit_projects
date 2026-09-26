import streamlit as st
import google.generativeai as genai
import requests
from PIL import Image
from io import BytesIO
from gtts import gTTS
import json
import os

st.set_page_config(page_title="AI Visual Novel", layout="wide")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@st.cache_resource
def load_model():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel("gemini-2.5-flash")

model = load_model()

st.sidebar.title("Story Settings")

genre = st.sidebar.selectbox(
    "Story Genre",
    [
        "Fantasy",
        "Sci-Fi",
        "Mystery",
        "Horror",
        "Adventure"
    ]
)

style = st.sidebar.selectbox(
    "Art Style",
    [
        "Anime",
        "Realistic",
        "Pixel Art",
        "Watercolor",
        "Dark Fantasy"
    ]
)

if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if "story_history" not in st.session_state:
    st.session_state.story_history = []

if "started" not in st.session_state:
    st.session_state.started = False


system_prompt = f"""
You are a Visual Novel AI.

Story Genre:
{genre}

Art Style:
{style}

Always reply ONLY in JSON.

Structure:

{{
"story_text":"Narration",
"image_prompt":"Detailed prompt for AI image generation in {style}",
"options":[
"choice1",
"choice2",
"choice3"
]
}}

Rules:

Return valid JSON only.

No markdown.

No explanation.

No ```json.

Image prompt must be highly detailed.
"""


def get_story(user_input):

    prompt = system_prompt + "\nPlayer Action: " + user_input

    response = st.session_state.chat.send_message(prompt)

    text = response.text.strip()

    text = text.replace("```json", "")
    text = text.replace("```", "")

    return json.loads(text)


def generate_image(prompt):

    url = "https://image.pollinations.ai/prompt/" + requests.utils.quote(prompt)

    response = requests.get(url, timeout=60)

    image = Image.open(BytesIO(response.content))

    return image


def generate_audio(text):

    tts = gTTS(text)

    filename = "narration.mp3"

    tts.save(filename)

    return filename


def render_scene(scene):

    st.markdown("## Story")

    st.write(scene["story_text"])

    try:

        image = generate_image(scene["image_prompt"])

        st.image(image, use_container_width=True)

    except:

        st.toast("Image server is busy, skipping visual...")

    try:

        audio = generate_audio(scene["story_text"])

        with open(audio, "rb") as file:

            st.audio(file.read(), format="audio/mp3")

    except:

        st.toast("Audio generation failed.")


st.title("🎮 AI Multi-Modal Visual Novel")

if not st.session_state.started:

    if st.button("Start Adventure"):

        first_scene = get_story("Begin the story")

        st.session_state.story_history.append(first_scene)

        st.session_state.started = True

        st.rerun()

else:

    current = st.session_state.story_history[-1]

    render_scene(current)

    st.markdown("---")

    st.subheader("Choose Your Next Action")

    for option in current["options"]:

        if st.button(option):

            next_scene = get_story(option)

            st.session_state.story_history.append(next_scene)

            st.rerun()

st.markdown("---")

with st.expander("Story History"):

    for index, scene in enumerate(st.session_state.story_history):

        st.markdown(f"### Scene {index+1}")

        st.write(scene["story_text"])