import streamlit as st
import cv2
import os
import numpy as np
import face_match  # Import our face matching module

# pip install streamlit opencv-python face-recognition numpy
# Run with: streamlit run sl.py

# Page configuration
st.set_page_config(layout="wide")
st.title("Real-time Face Recognition")

# Initialize session state for controlling the loop and caching last recognized face
if "run" not in st.session_state:
    st.session_state.run = True
if "last_recognized" not in st.session_state:
    st.session_state.last_recognized = None
if "missed_frames" not in st.session_state:
    st.session_state.missed_frames = 0

# Parameters: how many consecutive frames without detection before clearing cached data
MAX_MISSED_FRAMES = 15  # Adjust based on your needs

# Callback for the stop button
def stop_run():
    st.session_state.run = False

# Sidebar stop button (created only once)
st.sidebar.button("Stop", on_click=stop_run)

# Load known faces using our module function
known_face_encodings, known_face_names = face_match.load_known_faces("known_faces")

# Hardcoded data corresponding to each recognized face.
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

# Create two columns: one for status updates and one for the camera feed
col1, col2 = st.columns(2)
with col1:
    st.header("Status Updates")
    status_placeholder = st.empty()
with col2:
    st.header("Camera Feed")
    video_placeholder = st.empty()

# Initialize webcam
video_capture = cv2.VideoCapture(1)  # Adjust the camera index if needed

while st.session_state.run:
    ret, frame = video_capture.read()
    if not ret:
        st.error("Failed to access webcam")
        break

    # Process the current frame using the face_match module
    face_locations, face_labels, authenticated, alert_flag = face_match.process_frame(
        frame, known_face_encodings, known_face_names, match_tolerance=0.51
    )

    # Draw boxes and labels on the video frame
    for (top, right, bottom, left), name in zip(face_locations, face_labels):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Convert frame from BGR to RGB for Streamlit display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(rgb_frame, channels="RGB")

    # Determine if we should update our cached recognized face data:
    if authenticated:
        recognized_name = face_labels[0]
        st.session_state.last_recognized = recognized_name
        st.session_state.missed_frames = 0  # Reset counter on detection
    else:
        # If no face is authenticated, increase the missed frame counter
        st.session_state.missed_frames += 1
        # Clear the cached data only after MAX_MISSED_FRAMES are reached
        if st.session_state.missed_frames >= MAX_MISSED_FRAMES:
            st.session_state.last_recognized = None

    # Display corresponding data based on the cached recognized face
    if st.session_state.last_recognized:
        data_text = face_data_dict.get(
            st.session_state.last_recognized,
            f"<strong>{st.session_state.last_recognized}</strong> has no associated data."
        )
        blur_style = "filter: blur(5px);" if alert_flag else ""
        html_content = f"""
        <div style="font-size: 20px; {blur_style} background-color: #f0f0f0; padding: 10px;">
            {data_text}
        </div>
        """
        status_placeholder.markdown(html_content, unsafe_allow_html=True)
    else:
        status_placeholder.markdown("<p>No recognized face detected yet.</p>", unsafe_allow_html=True)

video_capture.release()
