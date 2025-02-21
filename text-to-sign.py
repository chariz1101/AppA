import streamlit as st
import os
from PIL import Image

# Streamlit app
st.title("Sign Language Image Display")

# Input sentence
sentence = st.text_input("Enter a sentence:")

# Path to the "handsigns" folder (adjust if needed)
handsigns_folder = "handsigns"

if sentence:
    image_paths = []
    for char in sentence.upper():
        if char.isalpha():
            image_path = os.path.join(handsigns_folder, f"{char}.png")
            if os.path.exists(image_path):
                image_paths.append(image_path)
            else:
                st.warning(f"Image not found for letter: {char}")
        elif char == " ":
            image_paths.append("space")
        elif char == ".":
            image_paths.append("stop")
        elif char == ",":
            image_paths.append("comma")
        elif char == "?":
            image_paths.append("question")
        elif char == "!":
            image_paths.append("exclamation")
        else:
            st.warning(f"Character '{char}' not supported.")

    if image_paths:
        st.write("## Sign Language Images")

        # Chunk the image paths into lines of max 10
        chunked_paths = [image_paths[i:i + 10] for i in range(0, len(image_paths), 10)]

        for chunk in chunked_paths:
            cols = st.columns(len(chunk))
            for i, path in enumerate(chunk):
                with cols[i]:
                    if path == "space":
                        st.write(" ")
                    elif path == "stop":
                        st.image(os.path.join(handsigns_folder, "stop.png"), width=150)
                    elif path == "comma":
                        st.image(os.path.join(handsigns_folder, "comma.png"), width=150)
                    elif path == "question":
                        st.image(os.path.join(handsigns_folder, "question.png"), width=150)
                    elif path == "exclamation":
                        st.image(os.path.join(handsigns_folder, "exclamation.png"), width=150)
                    else:
                        st.image(path, width=50)
            st.write("---")  # Separator between lines (optional)