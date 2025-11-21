"""
GROUP NAME: Team Makki
MEMBERS:
    Abegonia, Gil Marion
    Cabalan, Zachary Ezekiel
    Cerdeñola, Jance
    Cinco, Earl Seandrick
    Dimacali, Gyv Jered
    Lao, Keigan Iver

Project Honest Gaze
Partial Prototype: Eye Tracking Module
--------------------------------------
This script demonstrates the core functionality of the system —
real-time eye detection and tracking — using OpenCV and MediaPipe.

What it does:
- Activates the webcam
- Detects the user's face using MediaPipe Face Mesh
- Identifies the eye and iris landmark points
- Draws markers around the iris for visualization
- Computes normalized iris position inside the eye
- Determines and prints gaze direction (LEFT, RIGHT, UP, DOWN, CENTER)

Cheating-Related Detection Features:
- Long Gaze Detection:
    - Detects when the user looks away from the center for longer
      than a configurable threshold (default = 3 seconds).
    - After the threshold, the system issues a warning.
- Short Burst Detection:
    - Detects repeated short glances away from the center
      within a defined time window.
    - Uses a burst counter to track how many times the user looked away.
    - If the number of bursts exceeds the allowed limit
      (default = 3 bursts within 20 seconds), a warning is issued.

Configurable Parameters (for future proctor interface):
- Long gaze threshold (1–20 seconds)
- Burst window duration (5–60 seconds)
- Maximum allowed bursts (1–10 counts)

NOTE:
Required libraries are listed in requirements.txt.
Install them via: pip install -r requirements.txt
"""

import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
import time

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,   # Enables iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# ==============================
# Adjustable thresholds (defaults)
# ==============================
LONG_GAZE_THRESHOLD = 3      # Long look-away threshold (seconds)
BURST_MAX_COUNT = 3          # How many short bursts allowed
BURST_WINDOW = 20            # Time window for burst counting (seconds)
# ==============================

root = tk.Tk()
root.withdraw()

# Eye landmark indices (from MediaPipe FaceMesh)
RIGHT_IRIS = [468, 469, 470, 471]
RIGHT_EYE_LEFT = 33
RIGHT_EYE_RIGHT = 133
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

def get_average_iris_position(landmarks, indices, width, height):
    """Returns average (x, y) position of iris points."""
    xs = [landmarks[i].x * width for i in indices]
    ys = [landmarks[i].y * height for i in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)

# Long gaze timer
long_gaze_start = None

# Burst tracking
burst_timestamps = []

# Cooldown for warnings
last_warning_time = 0
WARNING_COOLDOWN = 3

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam error.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape
    current_time = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            iris_x, iris_y = get_average_iris_position(
                face_landmarks.landmark,
                RIGHT_IRIS,
                w, h
            )

            left_corner  = face_landmarks.landmark[RIGHT_EYE_LEFT].x * w
            right_corner = face_landmarks.landmark[RIGHT_EYE_RIGHT].x * w
            top_edge     = face_landmarks.landmark[RIGHT_EYE_TOP].y * h
            bottom_edge  = face_landmarks.landmark[RIGHT_EYE_BOTTOM].y * h

            horizontal_ratio = (iris_x - left_corner) / (right_corner - left_corner)
            vertical_ratio = (iris_y - top_edge) / (bottom_edge - top_edge)

            looking_away = horizontal_ratio < 0.35 or horizontal_ratio > 0.65

            # --------------------------
            # 1. LONG GAZE DETECTION
            # --------------------------
            if looking_away:
                if long_gaze_start is None:
                    long_gaze_start = current_time
                else:
                    if current_time - long_gaze_start >= LONG_GAZE_THRESHOLD:
                        if current_time - last_warning_time >= WARNING_COOLDOWN:
                            messagebox.showwarning(
                                "Long Gaze Warning",
                                "You have looked away from the screen for too long."
                            )
                            last_warning_time = current_time
                            long_gaze_start = None
            else:
                # Reset timer if user returns to center
                long_gaze_start = None

            # --------------------------
            # 2. SHORT BURST DETECTION
            # --------------------------
            if looking_away:
                # Add timestamp for this glance
                burst_timestamps.append(current_time)

            # Remove timestamps outside the burst window
            burst_timestamps = [
                t for t in burst_timestamps
                if current_time - t <= BURST_WINDOW
            ]

            # If bursts exceed allowed number
            if len(burst_timestamps) >= BURST_MAX_COUNT:
                if current_time - last_warning_time >= WARNING_COOLDOWN:
                    messagebox.showwarning(
                        "Burst Gaze Warning",
                        "You have been looking away repeatedly."
                    )
                    last_warning_time = current_time
                    burst_timestamps.clear()   # Reset after warning

            # Debug print
            print("Looking Away:", looking_away,
                  "Bursts:", len(burst_timestamps))

            # Draw iris for visualization
            cv2.circle(frame, (int(iris_x), int(iris_y)), 3, (0, 255, 0), -1)

    cv2.imshow("Gaze Direction Tracking - Honest Gaze", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
