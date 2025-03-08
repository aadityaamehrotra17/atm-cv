import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List
import time

# change push parameter
class PushDetector:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Parameters for push detection
        self.velocity_threshold = 10  # Adjust this value based on testing
        self.previous_positions: List[Tuple[float, float]] = []
        self.push_cooldown = 1.0  # Seconds between push detections
        self.last_push_time = 0
        
    def calculate_velocity(self, current_pos: Tuple[float, float], 
                         previous_pos: Tuple[float, float]) -> float:
        """Calculate the velocity between two positions"""
        dx = current_pos[0] - previous_pos[0]
        dy = current_pos[1] - previous_pos[1]
        return np.sqrt(dx*dx + dy*dy)
    
    def count_people(self, frame):
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, 
            min_detection_confidence=0.5
        )
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        if results.detections:
            return len(results.detections)
        return 0
    
    def detect_push(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect poses
        results = self.pose.process(frame_rgb)

        # Add after the line with self.pose.process(frame_rgb)
        person_count = self.count_people(frame)

        # Display person count on frame
        cv2.putText(frame, f"People: {person_count}", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if results.pose_landmarks:
            # Draw pose landmarks on frame
            self.mp_draw.draw_landmarks(
                frame, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS
            )
            
            # Get torso center point (average of shoulders)
            left_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            center_x = (left_shoulder.x + right_shoulder.x) / 2
            center_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Convert to pixel coordinates
            h, w, _ = frame.shape
            current_pos = (int(center_x * w), int(center_y * h))
            
            # Store position for velocity calculation
            self.previous_positions.append(current_pos)
            if len(self.previous_positions) > 5:  # Keep last 5 positions
                self.previous_positions.pop(0)
            
            # Calculate velocity if we have enough positions
            if len(self.previous_positions) >= 2:
                velocity = self.calculate_velocity(
                    self.previous_positions[-1],
                    self.previous_positions[-2]
                )
                
                # Check for push (sudden movement)
                current_time = time.time()
                if (velocity > self.velocity_threshold and 
                    current_time - self.last_push_time > self.push_cooldown and
                    person_count >= 2):
                    cv2.putText(frame, "PUSH DETECTED!", (50, 50),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.last_push_time = current_time
                
                # Draw velocity debug info
                cv2.putText(frame, f"Velocity: {velocity:.2f}", (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    detector = PushDetector()
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
            
        # Process frame
        frame = detector.detect_push(frame)
        
        # Display result
        cv2.imshow('Push Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
