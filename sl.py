import streamlit as st
import cv2
import os
import numpy as np
import time
import face_match  # Your face matching module
from streamlit_autorefresh import st_autorefresh

# pip install streamlit opencv-python face-recognition numpy streamlit-autorefresh
# Run with: streamlit run sl.py

# --- Page Configuration ---
st.set_page_config(page_title="VaultVision", layout="wide")
st.title("VaultVision")

# --- Initialization ---
if "mode" not in st.session_state:
    st.session_state.mode = "landing"  # Modes: landing, face_recognition, pin_entry, transaction, logout
if "current_user" not in st.session_state:
    st.session_state.current_user = None
if "last_seen" not in st.session_state:
    st.session_state.last_seen = time.time()

# Open the webcam only once and store it in session state
if "video_capture" not in st.session_state:
    st.session_state.video_capture = cv2.VideoCapture(1)

# Load known faces once and store in session state
if "known_face_encodings" not in st.session_state or "known_face_names" not in st.session_state:
    encodings, names = face_match.load_known_faces("known_faces")
    st.session_state.known_face_encodings = encodings
    st.session_state.known_face_names = names

# Hardcoded user PINs and account details
user_pins = {
    "nish": "1234",
    "arnav": "5678",
    "aadi": "9012"
}

face_data_dict = {
    "nish": (
        "<strong>Nish's Account:</strong><br>"
        "Balance: £5,000<br>"
        "Recent Transactions:<br>"
        "&nbsp;&nbsp;- £200 at Store A<br>"
        "&nbsp;&nbsp;- £150 at Store B"
    ),
    "arnav": (
        "<strong>Arnav's Account:</strong><br>"
        "Balance: £3,200<br>"
        "Recent Transactions:<br>"
        "&nbsp;&nbsp;- £120 at Cafe X<br>"
        "&nbsp;&nbsp;- £300 at Restaurant Y"
    ),
    "aadi": (
        "<strong>Aadi's Account:</strong><br>"
        "Balance: £7,800<br>"
        "Recent Transactions:<br>"
        "&nbsp;&nbsp;- £500 at Shop Z<br>"
        "&nbsp;&nbsp;- £230 at Grocery Q"
    )
}

# --- Layout: Two Columns ---
col_status, col_camera = st.columns(2)

# --- Auto-Refresh ---
st_autorefresh(interval=200, limit=10000, key="auto_refresh")

# --- Update Camera Feed in Right Column ---
ret, frame = st.session_state.video_capture.read()
if ret:
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    col_camera.image(rgb_frame, channels="RGB", use_container_width=True)
else:
    col_camera.error("Unable to capture video")

# --- Workflow in Left Column ---
if st.session_state.mode == "landing":
    col_status.header("Welcome to VaultVision")
    col_status.write("Click 'Start' to begin your transaction.")
    if col_status.button("Start", key="start_button"):
         st.session_state.mode = "face_recognition"

elif st.session_state.mode == "face_recognition":
    col_status.header("Face Recognition")
    col_status.write("Position your face in front of the camera.")
    ret, frame = st.session_state.video_capture.read()
    if ret:
        try:
            face_locations, face_labels, authenticated, alert_flag = face_match.process_frame(
                frame, st.session_state.known_face_encodings, st.session_state.known_face_names, match_tolerance=0.51
            )
        except Exception as e:
            st.error(f"Face recognition error: {e}")
            authenticated = False
        if authenticated:
            st.session_state.current_user = face_labels[0]
            st.session_state.last_seen = time.time()
            st.session_state.mode = "pin_entry"
        else:
            col_status.write("Face not recognized. Please try again.")
            if col_status.button("Back to Landing", key="back_button"):
                st.session_state.mode = "landing"
    else:
        col_status.write("Waiting for camera feed...")

elif st.session_state.mode == "pin_entry":
    col_status.header("PIN Entry")
    col_status.write(f"Hello, {st.session_state.current_user.capitalize()}! Please enter your PIN:")
    pin_input = col_status.text_input("PIN", type="password", key="pin_input")
    if col_status.button("Submit PIN", key="submit_pin"):
         if st.session_state.current_user in user_pins and pin_input == user_pins[st.session_state.current_user]:
              st.session_state.mode = "transaction"
         else:
              col_status.error("Incorrect PIN. Please try again.")

elif st.session_state.mode == "transaction":
    col_status.header("Account Details")
    details = face_data_dict.get(st.session_state.current_user, "No details available.")
    # Process a fresh frame to get alert flag
    ret, frame = st.session_state.video_capture.read()
    current_alert_flag = False
    if ret:
        try:
            _, face_labels_tx, authenticated_tx, alert_flag = face_match.process_frame(
                 frame, st.session_state.known_face_encodings, st.session_state.known_face_names, match_tolerance=0.51
            )
            current_alert_flag = alert_flag
        except Exception as e:
            current_alert_flag = False
    blur_style = "filter: blur(5px);" if current_alert_flag else ""
    col_status.markdown(f"<div style='{blur_style}'>{details}</div>", unsafe_allow_html=True)
    col_status.write("Keep your face in view. You will be logged out if no face is detected for 3 seconds.")
    if ret:
         try:
             _, face_labels_tx, authenticated_tx, _ = face_match.process_frame(
                 frame, st.session_state.known_face_encodings, st.session_state.known_face_names, match_tolerance=0.51
             )
         except Exception as e:
             authenticated_tx = False
         if authenticated_tx:
              st.session_state.last_seen = time.time()
         else:
              if time.time() - st.session_state.last_seen > 3:
                   st.session_state.mode = "logout"
    if col_status.button("Finish Transaction", key="finish_button"):
         st.session_state.mode = "logout"

elif st.session_state.mode == "logout":
    col_status.header("Thank You!")
    col_status.write("Thank you for using VaultVision. Please take your receipt and your card.")
    # Set logout_start if not already set.
    if "logout_start" not in st.session_state:
         st.session_state.logout_start = time.time()
    # If more than 3 seconds have passed, reset to landing.
    if time.time() - st.session_state.logout_start > 3:
         st.session_state.mode = "landing"
         st.session_state.current_user = None
         st.session_state.last_seen = time.time()
         del st.session_state.logout_start

# The auto-refresh component updates the UI every 200ms, so after 3 seconds in logout mode the landing page will appear.