import streamlit as st
import cv2
import face_recognition
import os
import numpy as np

# pip install streamlit opencv-python face-recognition numpy
# Run with: streamlit run sl.py

# Page config
st.set_page_config(layout="wide")
st.title("Real-time Face Recognition")

# Directory containing the known faces
KNOWN_FACES_DIR = "known_faces"

# Lists to store known face encodings and their corresponding names
known_face_encodings = []
known_face_names = []

# Load known faces
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
        else:
            st.warning(f"No face found in {filename}")

# Create two columns: status updates (left) and camera feed (right)
col1, col2 = st.columns(2)
with col1:
    st.header("Status Updates")
    status_placeholder = st.empty()
with col2:
    st.header("Camera Feed")
    video_placeholder = st.empty()

# Initialize webcam
video_capture = cv2.VideoCapture(1)  # Adjust camera index if needed

# Dummy data to display when the first face is authenticated
dummy_text = (
    "<strong>Account Balance:</strong> $5,000<br>"
    "<strong>Recent Transactions:</strong><br>"
    "&nbsp;&nbsp;- Transaction 1: $200<br>"
    "&nbsp;&nbsp;- Transaction 2: $150<br>"
    "<strong>Additional Info:</strong> Dummy data here."
)

while True:
    ret, frame = video_capture.read()
    if not ret:
        st.error("Failed to access webcam")
        break

    # Resize frame for faster processing and convert to RGB
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Determine labels for each detected face
    face_labels = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        face_labels.append(name)

    # Flag variables for authentication and alert
    authenticated = False
    alert_flag = False

    # The first detected face is considered "authenticated" if it is recognized
    if face_labels:
        if face_labels[0] != "Unknown":
            authenticated = True

    # If authenticated and more than one face is present, set alert flag
    if authenticated and len(face_encodings) > 1:
        alert_flag = True

    # Draw boxes and labels on the video frame (scale coordinates back to original size)
    for (top, right, bottom, left), name in zip(face_locations, face_labels):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Convert BGR frame to RGB for Streamlit display
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(rgb_frame, channels="RGB")

    # Update status column: if authenticated, display dummy data with blur if alert_flag is True.
    if authenticated:
        # Set blur style based on the alert flag.
        blur_style = "filter: blur(5px);" if alert_flag else ""
        html_content = f"""
        <div style="font-size: 20px; {blur_style} background-color: #f0f0f0; padding: 10px;">
            {dummy_text}
        </div>
        """
        status_placeholder.markdown(html_content, unsafe_allow_html=True)
    else:
        status_placeholder.markdown("<p>No recognized face detected yet.</p>", unsafe_allow_html=True)

    if not st.session_state.get('run', True):
        break

video_capture.release()
