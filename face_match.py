import cv2
import face_recognition
import os
import numpy as np

def load_known_faces(known_faces_dir="known_faces"):
    """
    Loads images from the given directory, computes face encodings,
    and returns lists of encodings and associated names.
    """
    known_face_encodings = []
    known_face_names = []
    for filename in os.listdir(known_faces_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(known_faces_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_face_encodings.append(encodings[0])
                # Use the file name (without extension) as the person's name
                name = os.path.splitext(filename)[0]
                known_face_names.append(name)
            else:
                print(f"No face found in {filename}")
    return known_face_encodings, known_face_names

def process_frame(frame, known_face_encodings, known_face_names, match_tolerance=0.51):
    """
    Processes a video frame:
      - Resizes the frame for faster processing and converts it to RGB.
      - Detects faces and computes encodings.
      - Matches detected faces to the known faces using the provided tolerance.
      - Determines if the first detected face is authenticated and sets an alert flag if extra faces are present.
      - Returns scaled face locations, face labels, authenticated flag, and alert flag.
    """
    # Resize frame for faster processing and convert to RGB
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and compute encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Determine labels for each detected face
    face_labels = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=match_tolerance)
        name = "Unknown"
        if matches and True in matches:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = int(np.argmin(face_distances))
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        face_labels.append(name)

    # Flag variables
    authenticated = bool(face_labels and face_labels[0] != "Unknown")
    alert_flag = authenticated and (len(face_encodings) > 1)

    # Scale face locations back to original frame size
    scaled_locations = []
    for (top, right, bottom, left) in face_locations:
        scaled_locations.append((top * 4, right * 4, bottom * 4, left * 4))

    return scaled_locations, face_labels, authenticated, alert_flag
