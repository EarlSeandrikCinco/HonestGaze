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
import time
import tkinter as tk
from tkinter import messagebox
from collections import deque
import winsound

# CONSTANTS
CALIBRATION_FRAMES = 40
SMOOTH_FACTOR = 0.25
WARNING_DELAY = 3            # seconds of continuous looking away -> long gaze
WARNING_COOLDOWN = 1         # cooldown between long-gaze warnings
PITCH_TOLERANCE = 10         # degrees tolerance for UP/DOWN detection

# Burst-glance detection (short glances definition)
BURST_WINDOW = 20            # seconds to look back
BURST_MIN_GLANCES = 3        # how many short glances required
SHORT_GLANCE_MIN = 0.75      # minimum duration for a short glance (blink-proof)
BURST_WARNING_COOLDOWN = 5  # seconds between burst warnings

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474, 475, 476, 477]

RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

RIGHT_EYE_LEFT = 33
RIGHT_EYE_RIGHT = 133

# For pitch estimation
NOSE_TIP = 1
CHIN = 152

# Tkinter setup for warnings
root = tk.Tk()
root.withdraw()


# Utility Functions
def get_average_iris_position(landmarks, indices, w, h):
    xs = [landmarks[i].x * w for i in indices]
    ys = [landmarks[i].y * h for i in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def estimate_head_pitch(landmarks, w, h):
    """Returns a rough pitch estimate in degrees. Positive ≈ looking down."""
    try:
        nose = landmarks[NOSE_TIP]
        chin = landmarks[CHIN]
        dy = (chin.y - nose.y) * h
        dx = (chin.x - nose.x) * w
        if abs(dx) < 1e-6:
            dx = 1e-6
        angle = abs(dy / dx)
        # scale to an approximate degree-like number, clamped
        return max(-45.0, min(45.0, angle * 30.0))
    except Exception:
        return 0.0


def draw_overlay(frame, lines):
    y = 30
    for text in lines:
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 25

def play_warning_sound():
    winsound.Beep(2000, 1200)

# MAIN PROGRAM
def main():
    cap = cv2.VideoCapture(0)

    neutral_vertical = None
    neutral_pitch = None
    prev_vertical_ratio = None
    neutral_queue = deque()

    gaze = "CENTER"
    looking_away = False
    last_gaze_direction = None

    look_start = None
    last_warning_time = 0

    # Burst glance tracking
    burst_events = deque()
    last_burst_warning = 0

    short_glance_min = SHORT_GLANCE_MIN         # blink-proof short glance definition
    burst_max_duration = WARNING_DELAY          # must be below long-gaze threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam error.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape
        current_time = time.time()

        status_lines = []

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Iris positions
            rx, ry = get_average_iris_position(face_landmarks, RIGHT_IRIS, w, h)
            lx, ly = get_average_iris_position(face_landmarks, LEFT_IRIS, w, h)

            # Eye vertical boundaries
            r_top = face_landmarks[RIGHT_EYE_TOP].y * h
            r_bottom = face_landmarks[RIGHT_EYE_BOTTOM].y * h
            l_top = face_landmarks[LEFT_EYE_TOP].y * h
            l_bottom = face_landmarks[LEFT_EYE_BOTTOM].y * h

            # Vertical ratios
            r_v = 1 - ((ry - r_top) / (r_bottom - r_top + 1e-6))
            l_v = 1 - ((ly - l_top) / (l_bottom - l_top + 1e-6))
            vertical_ratio = (r_v + l_v) / 2.0

            # Smooth vertical signal
            if prev_vertical_ratio is None:
                smoothed_vertical = vertical_ratio
            else:
                smoothed_vertical = (
                    SMOOTH_FACTOR * vertical_ratio +
                    (1 - SMOOTH_FACTOR) * prev_vertical_ratio
                )
            prev_vertical_ratio = smoothed_vertical

            # Head pitch estimation
            pitch = estimate_head_pitch(face_landmarks, w, h)

            # Pitch compensation
            vertical_corrected = smoothed_vertical - (pitch / 45.0)

            # CALIBRATION PHASE
            if neutral_vertical is None:
                neutral_queue.append((vertical_corrected, pitch))
                status_lines.append(
                    f"Calibrating... ({len(neutral_queue)}/{CALIBRATION_FRAMES})"
                )

                if len(neutral_queue) >= CALIBRATION_FRAMES:
                    neutral_vertical = sum(v for v, p in neutral_queue) / len(neutral_queue)
                    neutral_pitch = sum(p for v, p in neutral_queue) / len(neutral_queue)
                    print("Calibration complete.")
                    print("Neutral vertical:", neutral_vertical)
                    print("Neutral pitch:", neutral_pitch)

                gaze = "CENTER"
                looking_away = False

            else:
                # HORIZONTAL classification
                r_left = face_landmarks[RIGHT_EYE_LEFT].x * w
                r_right = face_landmarks[RIGHT_EYE_RIGHT].x * w
                horizontal_ratio = (rx - r_left) / (r_right - r_left + 1e-6)
                horizontal_ratio = max(0, min(horizontal_ratio, 1))

                deviation = vertical_corrected - neutral_vertical
                pitch_diff = abs(pitch - neutral_pitch)

                gaze = "CENTER"
                looking_away = False

                if horizontal_ratio < 0.40:
                    gaze = "LEFT"
                    looking_away = True

                elif horizontal_ratio > 0.60:
                    gaze = "RIGHT"
                    looking_away = True

                else:
                    # VERTICAL CLASSIFICATION (depends on head stability)
                    if pitch_diff <= PITCH_TOLERANCE:
                        if deviation < -0.12:
                            gaze = "UP"
                            looking_away = True
                        elif deviation > 0.20:
                            gaze = "DOWN"
                            looking_away = True

                # Save direction before warning trigger
                if gaze != "CENTER":
                    last_gaze_direction = gaze

                # Long-gaze timer with direction-specific warning
                if looking_away:
                    if look_start is None:
                        look_start = current_time
                    elif current_time - look_start >= WARNING_DELAY:
                        if current_time - last_warning_time >= WARNING_COOLDOWN:
                            direction_text = last_gaze_direction or "away"
                            play_warning_sound()
                            messagebox.showwarning(
                                "Gaze Warning",
                                f"You looked {direction_text} for too long.\n"
                                "Please refocus on the screen."
                            )

                            last_warning_time = current_time
                            look_start = None

                # BURST-GLANCE DETECTION
                if gaze == "CENTER":
                    if look_start is not None:
                        glance_duration = current_time - look_start

                        # must be real glance: blink-proof AND not long gaze
                        if short_glance_min <= glance_duration < WARNING_DELAY:
                            burst_events.append(current_time)

                        look_start = None

                    # remove old events
                    while burst_events and current_time - burst_events[0] > BURST_WINDOW:
                        burst_events.popleft()

                    # trigger burst warning
                    if (len(burst_events) >= BURST_MIN_GLANCES and
                            current_time - last_burst_warning >= BURST_WARNING_COOLDOWN):
                        play_warning_sound()
                        messagebox.showwarning(
                            "Suspicious Activity",
                            "Multiple quick suspicious glances detected.\n"
                            "Please keep your eyes on the screen."
                        )
                        last_burst_warning = current_time
                        burst_events.clear()

                status_lines.append(
                    f"h={horizontal_ratio:.2f} dev={deviation:.2f} pitch={pitch:.1f} diff={pitch_diff:.1f}"
                )

            # Draw debug iris markers
            cv2.circle(frame, (int(rx), int(ry)), 3, (0, 255, 0), -1)
            cv2.circle(frame, (int(lx), int(ly)), 3, (0, 255, 255), -1)

        else:
            status_lines.append("No face detected")

        # Overlay info
        status_lines.insert(0, f"Gaze: {gaze}")
        draw_overlay(frame, status_lines)

        cv2.imshow("Honest Gaze - Eye Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Entry point
if __name__ == "__main__":
    main()
