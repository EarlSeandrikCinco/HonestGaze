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
import time
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp

# ==============================
# Initialize MediaPipe Face Mesh
# ==============================
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,   # Enables IRIS tracking
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
WARNING_COOLDOWN = 3         # Cooldown between warnings
# ==============================

root = tk.Tk()
root.withdraw()

# ------------------------------------
# Correct iris and eye landmark indices
# ------------------------------------
RIGHT_IRIS = [469, 470, 471, 472]

RIGHT_EYE_LEFT = 263
RIGHT_EYE_RIGHT = 362
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

# Tracking variables
long_gaze_start = None
burst_timestamps = []
last_warning_time = 0

# Calibration storage
calib_samples = []
CALIBRATION_FRAMES = 60   # ~1–2 seconds
calibrated = False

center_h = 0.5  # Default values (in case calibration fails)
center_v = 0.5


# ==============================
# Classification Function
# ==============================
def classify_gaze(hor, ver):
    """
    Returns LEFT, RIGHT, UP, DOWN, CENTER
    Uses realistic thresholds after calibration.
    """

    # Horizontal classification
    if hor < 0.25:
        h_dir = "LEFT"
    elif hor > 0.75:
        h_dir = "RIGHT"
    else:
        h_dir = "CENTER"

    # Vertical classification
    if ver < 0.30:
        v_dir = "UP"
    elif ver > 0.70:
        v_dir = "DOWN"
    else:
        v_dir = "CENTER"

    # Combined
    if h_dir == "CENTER" and v_dir == "CENTER":
        return "CENTER"
    elif h_dir == "CENTER":
        return v_dir
    elif v_dir == "CENTER":
        return h_dir
    else:
        return f"{h_dir} & {v_dir}"


# ==============================
# Iris Position Helper
# ==============================
def get_average_iris_position(landmarks, indices, width, height):
    xs = [landmarks[i].x * width for i in indices]
    ys = [landmarks[i].y * height for i in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


# ==============================
# Main Program
# ==============================
def main():
    global long_gaze_start, burst_timestamps, last_warning_time
    global calibrated, center_h, center_v

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam error.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        h, w, _ = frame.shape
        current_time = time.time()
        gaze_direction = "CENTER"  # default

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                # ---- Iris Position ----
                iris_x, iris_y = get_average_iris_position(
                    face_landmarks.landmark, RIGHT_IRIS, w, h
                )

                # ---- Eye boundaries ----
                left_corner = face_landmarks.landmark[RIGHT_EYE_LEFT].x * w
                right_corner = face_landmarks.landmark[RIGHT_EYE_RIGHT].x * w

                top_edge = face_landmarks.landmark[RIGHT_EYE_TOP].y * h
                bottom_edge = face_landmarks.landmark[RIGHT_EYE_BOTTOM].y * h

                # ---- Ratios ----
                horizontal_ratio = (iris_x - left_corner) / (right_corner - left_corner)
                vertical_ratio = (iris_y - top_edge) / (bottom_edge - top_edge)

                horizontal_ratio = max(0, min(horizontal_ratio, 1))
                vertical_ratio = max(0, min(vertical_ratio, 1))

                # ===========================================
                # AUTO CALIBRATION (first 1–2 seconds)
                # ===========================================
                if not calibrated:
                    calib_samples.append((horizontal_ratio, vertical_ratio))

                    if len(calib_samples) >= CALIBRATION_FRAMES:
                        center_h = sum(x for x, y in calib_samples) / len(calib_samples)
                        center_v = sum(y for x, y in calib_samples) / len(calib_samples)
                        calibrated = True
                        print("CENTER CALIBRATED:", center_h, center_v)

                else:
                    # Convert raw ratios to relative (center = 0.5)
                    horizontal_relative = horizontal_ratio - center_h
                    vertical_relative = vertical_ratio - center_v

                    # Normalize into 0–1 values for classification
                    hor = 0.5 + horizontal_relative * 2
                    ver = 0.5 + vertical_relative * 2

                    hor = max(0, min(hor, 1))
                    ver = max(0, min(ver, 1))

                    gaze_direction = classify_gaze(hor, ver)

                print("Gaze:", gaze_direction)

                # === Flag if user is not looking at center ===
                looking_away = gaze_direction != "CENTER"

                # ======================================================
                # 1. LONG GAZE DETECTION
                # ======================================================
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
                    long_gaze_start = None

                # ======================================================
                # 2. BURST GAZE DETECTION
                # ======================================================
                if looking_away:
                    burst_timestamps.append(current_time)

                burst_timestamps = [
                    t for t in burst_timestamps
                    if current_time - t <= BURST_WINDOW
                ]

                if len(burst_timestamps) >= BURST_MAX_COUNT:
                    if current_time - last_warning_time >= WARNING_COOLDOWN:
                        messagebox.showwarning(
                            "Burst Gaze Warning",
                            "You have been looking away repeatedly."
                        )
                        last_warning_time = current_time
                        burst_timestamps.clear()

                print("Away:", looking_away, "Bursts:", len(burst_timestamps))

                cv2.circle(frame, (int(iris_x), int(iris_y)), 3, (0, 255, 0), -1)

        cv2.imshow("Gaze Direction Tracking - Honest Gaze", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
