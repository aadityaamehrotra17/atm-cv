import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List
import time
from threading import Thread
import queue

# Set page config
st.set_page_config(layout="wide", page_title="Push Detection System")

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
        self.status_info = {
            "velocity": 0.0,
            "people": 0,
            "push_detected": False
        }
        
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

        # Count people in the frame
        person_count = self.count_people(frame)
        self.status_info["people"] = person_count

        # Display person count on frame (we'll hide this in the Streamlit version)
        # cv2.putText(frame, f"People: {person_count}", (50, 150),
        #         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
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
                
                self.status_info["velocity"] = velocity
                
                # Check for push (sudden movement)
                current_time = time.time()
                if (velocity > self.velocity_threshold and 
                    current_time - self.last_push_time > self.push_cooldown and
                    person_count >= 2):
                    self.status_info["push_detected"] = True
                    # cv2.putText(frame, "PUSH DETECTED!", (50, 50),
                    #           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    self.last_push_time = current_time
                else:
                    self.status_info["push_detected"] = False
                
                # Draw velocity debug info on frame (we'll hide this in the Streamlit version)
                # cv2.putText(frame, f"Velocity: {velocity:.2f}", (50, 100),
                #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return frame

def video_stream_thread(detector, frame_queue, status_queue):
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read from webcam")
            break
            
        # Process frame
        processed_frame = detector.detect_push(frame)
        
        # Send processed frame to main thread
        if not frame_queue.full():
            frame_queue.put(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            
        # Send status info to main thread
        if not status_queue.full():
            status_queue.put(detector.status_info.copy())
            
        # Break if q is pressed (not applicable in Streamlit but kept for debugging)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

def main():
    # Header
    st.title("Push Detection System")
    st.markdown("### Monitor and detect pushing incidents")
    
    # Setup layout with two columns
    col1, col2 = st.columns(2)
    
    # Status panel in left column
    with col1:
        st.subheader("Status Panel")
        
        # Create placeholders for metrics
        people_counter = st.empty()
        velocity_display = st.empty()
        
        # Create placeholder for push alert
        push_alert = st.empty()
    
    # Video feed in right column
    with col2:
        st.subheader("Video Feed")
        video_frame = st.empty()
    
    # Create detector and communication queues
    detector = PushDetector()
    frame_queue = queue.Queue(maxsize=1)
    status_queue = queue.Queue(maxsize=1)
    
    # Start video thread
    thread = Thread(target=video_stream_thread, args=(detector, frame_queue, status_queue))
    thread.daemon = True
    thread.start()
    
    # Main loop
    try:
        alert_start_time = 0  # Track when the alert started
        showing_alert = False  # Track if we're currently showing an alert
        while True:
            # Update video frame if available
            if not frame_queue.empty():
                video_frame.image(frame_queue.get())
                
            # Update status if available
            if not status_queue.empty():
                status = status_queue.get()
                
                # Update metrics
                people_counter.metric("People Detected", status["people"])
                velocity_display.metric("Movement Velocity", f"{status['velocity']:.2f}")
                
                # Show push alert
                current_time = time.time()
                if status["push_detected"]:
                    if not showing_alert:
                        push_alert.error("⚠️ PUSH DETECTED! ⚠️")
                        alert_start_time = current_time
                        showing_alert = True
                    
                else:
                    if showing_alert and (current_time - alert_start_time >= 3.0):
                        push_alert.empty()
                        showing_alert = False
                    
            # Short sleep to prevent high CPU usage
            time.sleep(0.01)
                
    except Exception as e:
        st.error(f"Error: {e}")

if __name__ == "__main__":
    main()