import streamlit as st
import cv2
import face_recognition
import os
from PIL import Image
import numpy as np

# pip install streamlit opencv-python face-recognition pillow numpy
# streamlit run sl.py

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
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
        else:
            st.warning(f"No face found in {filename}")

# Create two columns
col1, col2 = st.columns(2)

# Status updates column
with col1:
    st.header("Status Updates")
    status_placeholder = st.empty()

# Camera feed column
with col2:
    st.header("Camera Feed")
    video_placeholder = st.empty()

# Initialize webcam
video_capture = cv2.VideoCapture(1)

while True:
    ret, frame = video_capture.read()
    if not ret:
        st.error("Failed to access webcam")
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Initialize status text
    status_text = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if len(known_face_encodings) > 0:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = face_distances.argmin()
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        # Scale face locations back
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        # Add to status updates
        status_text.append(f"Detected: {name}")

    # Convert BGR to RGB for Streamlit
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Update the video feed
    video_placeholder.image(rgb_frame, channels="RGB")
    
    # Update status
    if status_text:
        status_placeholder.write("\n".join(status_text))
    else:
        status_placeholder.write("No faces detected")

    # Check for streamlit stop
    if not st.session_state.get('run', True):
        break

# Release resources
video_capture.release()