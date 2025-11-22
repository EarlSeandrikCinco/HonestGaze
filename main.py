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
from collections import deque
import numpy as np

# -----------------------------
# MediaPipe Face Mesh
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# -----------------------------
# Eye landmarks
# -----------------------------
RIGHT_IRIS = [468, 469, 470, 471]
LEFT_IRIS = [473, 474, 475, 476]

RIGHT_EYE_LEFT = 33
RIGHT_EYE_RIGHT = 133
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145

LEFT_EYE_LEFT = 362
LEFT_EYE_RIGHT = 263
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

# Head pose landmarks (3D reference)
HEAD_POINTS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "mouth_left": 61,
    "mouth_right": 291
}

# 3D model points in mm (approximate)
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],        # nose tip
    [0.0, -63.6, -12.5],    # chin
    [-43.3, 32.7, -26.0],   # left eye corner
    [43.3, 32.7, -26.0],    # right eye corner
    [-28.9, -28.9, -24.1],  # mouth left
    [28.9, -28.9, -24.1]    # mouth right
], dtype=np.float64)

# -----------------------------
# Tkinter
# -----------------------------
root = tk.Tk()
root.withdraw()

# -----------------------------
# Timed gaze tracking
# -----------------------------
look_start = None
WARNING_DELAY = 2 # feels more like 3 seconds when set to 2 instead of 3
last_warning_time = 0
WARNING_COOLDOWN = 3

neutral_queue = deque(maxlen=50)
neutral_vertical = None
prev_vertical_ratio = None
SMOOTH_FACTOR = 0.3
NEUTRAL_UPDATE_FACTOR = 0.01

def get_average_iris_position(landmarks, indices, width, height):
    xs = [landmarks[i].x * width for i in indices]
    ys = [landmarks[i].y * height for i in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)

def estimate_head_pose(landmarks, w, h):
    image_points = np.array([
        (landmarks[HEAD_POINTS["nose_tip"]].x * w,
         landmarks[HEAD_POINTS["nose_tip"]].y * h),
        (landmarks[HEAD_POINTS["chin"]].x * w,
         landmarks[HEAD_POINTS["chin"]].y * h),
        (landmarks[HEAD_POINTS["left_eye_corner"]].x * w,
         landmarks[HEAD_POINTS["left_eye_corner"]].y * h),
        (landmarks[HEAD_POINTS["right_eye_corner"]].x * w,
         landmarks[HEAD_POINTS["right_eye_corner"]].y * h),
        (landmarks[HEAD_POINTS["mouth_left"]].x * w,
         landmarks[HEAD_POINTS["mouth_left"]].y * h),
        (landmarks[HEAD_POINTS["mouth_right"]].x * w,
         landmarks[HEAD_POINTS["mouth_right"]].y * h)
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1))  # assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0

    rmat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat([rmat, translation_vector])
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch = euler_angles[0, 0]  # rotation around x-axis
    return pitch  # in degrees

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

            # --- Iris positions ---
            rx, ry = get_average_iris_position(face_landmarks.landmark, RIGHT_IRIS, w, h)
            lx, ly = get_average_iris_position(face_landmarks.landmark, LEFT_IRIS, w, h)

            # --- Eye boundaries ---
            r_top = face_landmarks.landmark[RIGHT_EYE_TOP].y * h
            r_bottom = face_landmarks.landmark[RIGHT_EYE_BOTTOM].y * h
            l_top = face_landmarks.landmark[LEFT_EYE_TOP].y * h
            l_bottom = face_landmarks.landmark[LEFT_EYE_BOTTOM].y * h

            # --- Vertical ratio ---
            r_vertical_ratio = 1 - (ry - r_top) / (r_bottom - r_top)
            l_vertical_ratio = 1 - (ly - l_top) / (l_bottom - l_top)
            vertical_ratio = (r_vertical_ratio + l_vertical_ratio) / 2

            # --- Smooth ---
            if prev_vertical_ratio is None:
                smoothed_vertical = vertical_ratio
            else:
                smoothed_vertical = SMOOTH_FACTOR * vertical_ratio + (1 - SMOOTH_FACTOR) * prev_vertical_ratio
            prev_vertical_ratio = smoothed_vertical

            # --- Head pose compensation ---
            pitch = estimate_head_pose(face_landmarks.landmark, w, h)
            vertical_ratio_corrected = smoothed_vertical - (pitch / 90.0)  # scale pitch (-90 to 90) to ratio

            # --- Neutral calibration ---
            if neutral_vertical is None:
                neutral_queue.append(vertical_ratio_corrected)
                if len(neutral_queue) == neutral_queue.maxlen:
                    neutral_vertical = sum(neutral_queue) / len(neutral_queue)
                    print("Neutral vertical set to:", neutral_vertical)
            else:
                r_left = face_landmarks.landmark[RIGHT_EYE_LEFT].x * w
                r_right = face_landmarks.landmark[RIGHT_EYE_RIGHT].x * w
                horizontal_ratio = (rx - r_left) / (r_right - r_left)
                deviation = vertical_ratio_corrected - neutral_vertical
                if 0.4 <= horizontal_ratio <= 0.6 and -0.05 <= deviation <= 0.05:
                    neutral_vertical += (vertical_ratio_corrected - neutral_vertical) * NEUTRAL_UPDATE_FACTOR

            # --- Horizontal ratio ---
            r_left = face_landmarks.landmark[RIGHT_EYE_LEFT].x * w
            r_right = face_landmarks.landmark[RIGHT_EYE_RIGHT].x * w
            horizontal_ratio = (rx - r_left) / (r_right - r_left)

            # --- Determine gaze ---
            gaze = "CENTER"
            looking_away = False

            if horizontal_ratio < 0.40:
                gaze = "LEFT"
                looking_away = True
            elif horizontal_ratio > 0.60:
                gaze = "RIGHT"
                looking_away = True
            else:
                if neutral_vertical is not None:
                    deviation = vertical_ratio_corrected - neutral_vertical
                    if deviation < -0.10:
                        gaze = "UP"
                        looking_away = True
                    elif deviation > 0.15:
                        gaze = "DOWN"
                        looking_away = True

            # --- Timed warning ---
            if looking_away:
                if look_start is None:
                    look_start = current_time
                elif current_time - look_start >= WARNING_DELAY:
                    if current_time - last_warning_time >= WARNING_COOLDOWN:
                        messagebox.showwarning(
                            "Long Glance Warning",
                            "Please focus your gaze on the center of the screen!"
                        )
                        last_warning_time = current_time
                        look_start = None
            else:
                look_start = None

            print(gaze)	
            cv2.circle(frame, (int(rx), int(ry)), 3, (0, 255, 0), -1)
            cv2.circle(frame, (int(lx), int(ly)), 3, (0, 255, 255), -1)

    cv2.imshow("Gaze Direction Tracking - Honest Gaze", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
