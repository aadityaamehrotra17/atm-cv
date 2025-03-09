import cv2
import face_recognition
import numpy as np
import math

def detect_wink_vectorized(frame, ear_threshold=0.21):
    """
    Given a frame, returns True if a wink is detected using vectorized operations.
    
    The function:
      - Converts the frame to RGB.
      - Uses face_recognition to extract facial landmarks for all detected faces.
      - Computes the eye aspect ratio (EAR) for both eyes for each face using vectorized NumPy operations.
      - Returns True if, for any face, one eye's EAR is below the threshold while the other is above.
    """
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    landmarks_list = face_recognition.face_landmarks(rgb_frame)
    
    if not landmarks_list:
        return False

    # We'll collect EARs for each face in a list
    ear_list = []
    for landmarks in landmarks_list:
        # Convert eye landmarks to numpy arrays
        left_eye = np.array(landmarks.get("left_eye"))
        right_eye = np.array(landmarks.get("right_eye"))
        if left_eye is None or right_eye is None or len(left_eye) < 6 or len(right_eye) < 6:
            continue

        # Compute EAR for left eye using vectorized operations
        left_A = np.linalg.norm(left_eye[1] - left_eye[5])
        left_B = np.linalg.norm(left_eye[2] - left_eye[4])
        left_C = np.linalg.norm(left_eye[0] - left_eye[3])
        left_ear = (left_A + left_B) / (2.0 * left_C) if left_C != 0 else 0

        # Compute EAR for right eye using vectorized operations
        right_A = np.linalg.norm(right_eye[1] - right_eye[5])
        right_B = np.linalg.norm(right_eye[2] - right_eye[4])
        right_C = np.linalg.norm(right_eye[0] - right_eye[3])
        right_ear = (right_A + right_B) / (2.0 * right_C) if right_C != 0 else 0

        ear_list.append([left_ear, right_ear])
    
    if not ear_list:
        return False

    # Convert the list to a NumPy array of shape (N, 2) where N is number of faces
    ears = np.array(ear_list)  # Each row is [left_ear, right_ear]
    # Compute a boolean array: True if left eye is below threshold and right eye is above, or vice versa.
    wink_flags = ((ears[:, 0] < ear_threshold) & (ears[:, 1] >= ear_threshold)) | \
                 ((ears[:, 1] < ear_threshold) & (ears[:, 0] >= ear_threshold))
    return np.any(wink_flags)
