import streamlit as st
import pickle
import time
import mediapipe as mp
import numpy as np
import os
import cv2
import threading
from PIL import Image

def text_to_sign_page():
    st.title("Text to Sign Page")
    sentence = st.text_input("Enter a sentence:")
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
                st.write("---")

def sign_to_text_page():
    st.title("Sign to Text Page")
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)
    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 30: 'Erase', 31: 'Space', 32: 'Stop'}
    video_display = st.image([], width=640)
    detected_letters_display = st.empty()
    detection_status_display = st.empty()
    detected_letters = ""
    current_letter = None
    letter_start_time = None
    cooldown_end_time = None
    cap = cv2.VideoCapture(0)

    def process_frame():
        nonlocal detected_letters, current_letter, letter_start_time, cooldown_end_time
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            H, W, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                               mp_drawing_styles.get_default_hand_landmarks_style(),
                                               mp_drawing_styles.get_default_hand_connections_style())
                    x_ = []
                    y_ = []
                    data_aux = []
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        x_.append(x)
                        y_.append(y)
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))
                    x1 = int(min(x_) * W) - 10
                    y1 = int(min(y_) * H) - 10
                    x2 = int(max(x_) * W) + 10
                    y2 = int(max(y_) * H) + 10
                    if data_aux:
                        try:
                            prediction = model.predict([np.asarray(data_aux)])
                            predicted_character = labels_dict[int(prediction[0])]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                                        (0, 0, 0), 3, cv2.LINE_AA)
                            current_time = time.time()
                            if current_letter != predicted_character:
                                current_letter = predicted_character
                                letter_start_time = current_time
                                cooldown_end_time = None
                                detection_status_display.text(f"Recognizing: {current_letter}")
                            elif letter_start_time is not None and current_time - letter_start_time >= 5:
                                if cooldown_end_time is None:
                                    cooldown_end_time = current_time + 3
                                    detection_status_display.text(f"Cooling down: {current_letter}")
                                elif current_time >= cooldown_end_time:
                                    if current_letter == "Erase":
                                        detected_letters = detected_letters[:-1]
                                        detection_status_display.text("Erased last letter")
                                    elif current_letter == "Space":
                                        detected_letters += " "
                                        detection_status_display.text("Added space")
                                    elif current_letter == "Stop":
                                        detected_letters += "."
                                        detection_status_display.text("Sentence finished")
                                    elif current_letter not in ["Erase", "Space", "Stop"]:
                                        detected_letters += current_letter
                                        detection_status_display.text(f"Letter {current_letter} added")
                                    current_letter = None
                                    cooldown_end_time = None
                                    letter_start_time = None
                        except ValueError as e:
                            print(f"ValueError during prediction: {e}")
                            print(f"data_aux length: {len(data_aux)}")
            video_display.image(frame, channels="BGR")
            detected_letters_display.text("Detected Letters: " + detected_letters)

    threading.Thread(target=process_frame).start()

    st.write("Press 'q' in the video feed to stop.")

    def stop_camera():
      cap.release()
      cv2.destroyAllWindows()

    st.button("Stop Camera", on_click=stop_camera)

page = st.sidebar.selectbox("Go to", ("Text to Sign", "Sign to Text"))
if page == "Text to Sign":
    text_to_sign_page()
elif page == "Sign to Text":
    sign_to_text_page()