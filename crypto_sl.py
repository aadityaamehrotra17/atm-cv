import streamlit as st
import cv2
import os
import numpy as np
import face_match  # Import our face matching module
from cryptography.fernet import Fernet

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

# Parameter: number of consecutive frames to wait before clearing cached data
MAX_MISSED_FRAMES = 15

# Callback for the stop button
def stop_run():
    st.session_state.run = False

# Sidebar stop button (created only once)
st.sidebar.button("Stop", on_click=stop_run)

# Load known faces using our module function
known_face_encodings, known_face_names = face_match.load_known_faces("known_faces")

# Set up encryption using Fernet
key = os.environ.get("FIN_DATA_KEY")
if not key:
    raise ValueError("FIN_DATA_KEY environment variable not found. Please set it to your encryption key.")
fernet = Fernet(key.encode())

# Hardcoded plain text records for each recognized face (these will be stored encrypted)
records_plain = {
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


# Encrypt the records and store in the dictionary
face_data_dict = {}
for name, text in records_plain.items():
    encrypted_text = fernet.encrypt(text.encode())
    face_data_dict[name] = encrypted_text

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

    # Process the frame using the face_match module
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

    # Update cached recognized face or increase missed frame counter
    if authenticated:
        recognized_name = face_labels[0]
        st.session_state.last_recognized = recognized_name
        st.session_state.missed_frames = 0
    else:
        st.session_state.missed_frames += 1
        if st.session_state.missed_frames >= MAX_MISSED_FRAMES:
            st.session_state.last_recognized = None

    # Display corresponding decrypted data based on the cached recognized face
    if st.session_state.last_recognized:
        # Retrieve the encrypted record and decrypt it
        encrypted_data = face_data_dict.get(
            st.session_state.last_recognized,
            None
        )
        if encrypted_data:
            try:
                decrypted_data = fernet.decrypt(encrypted_data).decode()
            except Exception as e:
                decrypted_data = "Error decrypting data."
        else:
            decrypted_data = f"<strong>{st.session_state.last_recognized}</strong> has no associated data."

        # Apply blur style if alert_flag is true (multiple faces detected)
        blur_style = "filter: blur(5px);" if alert_flag else ""
        html_content = f"""
        <div style="font-size: 20px; {blur_style} background-color: #f0f0f0; padding: 10px;">
            {decrypted_data}
        </div>
        """
        status_placeholder.markdown(html_content, unsafe_allow_html=True)
    else:
        status_placeholder.markdown("<p>No recognized face detected yet.</p>", unsafe_allow_html=True)

video_capture.release()
