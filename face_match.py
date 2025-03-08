import cv2
import face_recognition
import os

# Directory containing the known faces
KNOWN_FACES_DIR = "known_faces"

# Lists to store known face encodings and their corresponding names
known_face_encodings = []
known_face_names = []

# Loop over the images in the known faces directory
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        # Construct the full path to the image file
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        
        # Load the image using face_recognition
        image = face_recognition.load_image_file(image_path)
        
        # Get face encodings; assume each image has one face
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            
            # Extract the name from the filename (e.g., "Alice.jpg" -> "Alice")
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
        else:
            print(f"No face found in {filename}")

# Initialize webcam video capture
video_capture = cv2.VideoCapture(1) #which camera to use

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing and convert to RGB
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and get their encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare detected face to known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin() if face_distances.size > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Scale face locations back to the original frame size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle and label around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Display the live video feed
    cv2.imshow("Live Face Recognition", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
