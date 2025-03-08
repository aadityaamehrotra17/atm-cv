import cv2
import face_recognition
import os
import numpy as np

# Directory containing the known faces
KNOWN_FACES_DIR = "known_faces"

# Lists to store known face encodings and their corresponding names
known_face_encodings = []
known_face_names = []

# Load known faces from the folder
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            # Use the file name (without extension) as the person's name
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
        else:
            print(f"No face found in {filename}")

# Initialize webcam video capture
video_capture = cv2.VideoCapture(1)  # Adjust the camera index if needed

# Parameters for stricter matching
MATCH_TOLERANCE = 0.51  # Lower tolerance for stricter matching

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing and convert to RGB
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and compute encodings for all detected faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    # Determine labels for each detected face
    face_labels = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=MATCH_TOLERANCE)
        name = "Unknown"
        if True in matches:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
        face_labels.append(name)

    # Flag variables:
    authenticated = False
    alert_flag = False

    # Consider the first detected face as the "authenticated" one if recognized
    if face_labels:
        if face_labels[0] != "Unknown":
            authenticated = True

    # If authenticated and there are additional faces, set the alert flag
    if authenticated and len(face_encodings) > 1:
        alert_flag = True

    # Draw boxes and labels for all detected faces (scale coordinates back to original frame)
    for (top, right, bottom, left), name in zip(face_locations, face_labels):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display an alert message if the alert flag is set
    if alert_flag:
        cv2.putText(frame, "ALERT!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
