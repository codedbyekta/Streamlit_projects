import streamlit as st
import requests
import random

st.set_page_config(page_title="AI Image Studio")

st.title(" The AI Image Studio")

st.sidebar.header("Generation Settings")

art_style = st.sidebar.selectbox(
    "Select Art Style",
    [
        "Photorealistic",
        "Anime",
        "Vintage Victorian",
        "Sketch",
        "3D Render"
    ]
)

width = st.sidebar.slider(
    "Image Width",
    min_value=256,
    max_value=1024,
    value=768,
    step=64
)

height = st.sidebar.slider(
    "Image Height",
    min_value=256,
    max_value=1024,
    value=768,
    step=64
)

magic_enhance = st.sidebar.checkbox(" Enable Magic Enhance")

user_prompt = st.text_input("Describe your masterpiece:")

surprise_prompts = [
    "An astronaut riding a horse on Mars",
    "A cyberpunk street food vendor in Tokyo",
    "A panda coding inside Google headquarters",
    "A floating castle above the clouds",
    "A dragon drinking coffee in a futuristic cafe"
]

col1, col2 = st.columns(2)

with col1:
    generate = st.button(" Generate Image")

with col2:
    surprise = st.button(" Surprise Me!")

if generate:

    if user_prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:

        full_prompt = f"{user_prompt}, make the art style: {art_style}"

        if magic_enhance:
            full_prompt += ", masterpiece, 8k resolution, highly detailed, trending on artstation, unreal engine 5 render"

        url = f"https://image.pollinations.ai/prompt/{full_prompt}?width={width}&height={height}"

        with st.spinner("Rendering the image..."):

            response = requests.get(url)

            if response.status_code == 200:

                st.success("Image Generated")

                st.image(
                    response.content,
                    caption=full_prompt,
                    use_container_width=True
                )

                st.download_button(
                    label=" Download Image",
                    data=response.content,
                    file_name=f"{art_style}_image.png",
                    mime="image/png"
                )

            else:
                st.error("API is not working")

if surprise:

    random_prompt = random.choice(surprise_prompts)

    st.info(f" Prompt: {random_prompt}")

    full_prompt = f"{random_prompt}, make the art style: {art_style}"

    if magic_enhance:
        full_prompt += ", masterpiece, 8k resolution, highly detailed, trending on artstation, unreal engine 5 render"

    url = f"https://image.pollinations.ai/prompt/{full_prompt}?width={width}&height={height}"

    with st.spinner("Generating Surprise Image..."):

        response = requests.get(url)

        if response.status_code == 200:

            st.success("Surprise Image Generated!")

            st.image(
                response.content,
                caption=random_prompt,
                use_container_width=True
            )

            st.download_button(
                label=" Download Surprise Image",
                data=response.content,
                file_name=f"{art_style}_surprise.png",
                mime="image/png"
            )

        else:
            st.error("API is not working")