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
- Identifies the eye landmark points
- Draws markers around the eyes for visualization
- Prints coordinates (for future logic such as gaze direction)
- Detects Gaze Direction:
    - Detects the face using MediaPipe Face Mesh
    - Locates the iris landmarks
    - Compares iris position inside the eye to determine gaze direction
    - Prints events (LEFT, RIGHT, UP, DOWN, CENTER)

NOTE:
Required libraries are listed in requirements.txt.
Install them via: pip install -r requirements.txt
"""

import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,   # Enables iris tracking
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Eye landmark indices (from MediaPipe FaceMesh)
RIGHT_IRIS = [468, 469, 470, 471]
RIGHT_EYE_LEFT = 33    # Outer corner of right eye
RIGHT_EYE_RIGHT = 133  # Inner corner of right eye
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145


def get_average_iris_position(landmarks, indices, width, height):
    """Returns average (x, y) position of iris points."""
    xs = [landmarks[i].x * width for i in indices]
    ys = [landmarks[i].y * height for i in indices]
    return sum(xs) / len(xs), sum(ys) / len(ys)


while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam error.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # 1. Get iris position
            iris_x, iris_y = get_average_iris_position(
                face_landmarks.landmark,
                RIGHT_IRIS,
                w, h
            )

            # 2. Get eye boundary positions
            left_corner  = face_landmarks.landmark[RIGHT_EYE_LEFT].x * w
            right_corner = face_landmarks.landmark[RIGHT_EYE_RIGHT].x * w
            top_edge     = face_landmarks.landmark[RIGHT_EYE_TOP].y * h
            bottom_edge  = face_landmarks.landmark[RIGHT_EYE_BOTTOM].y * h

            # 3. Normalize iris position inside the eye (0 to 1)
            horizontal_ratio = (iris_x - left_corner) / (right_corner - left_corner)
            vertical_ratio = (iris_y - top_edge) / (bottom_edge - top_edge)

            gaze = ""

            # 4. Determine LEFT–RIGHT
            if horizontal_ratio < 0.35:
                gaze = "Looking LEFT"
            elif horizontal_ratio > 0.65:
                gaze = "Looking RIGHT"
            else:
                gaze = "Centered horizontally"

            # 5. Determine UP–DOWN
            if vertical_ratio < 0.35:
                gaze += " & UP"
            elif vertical_ratio > 0.65:
                gaze += " & DOWN"
            else:
                gaze += " & CENTER V"

            # Print event output
            print(gaze)

            # Draw iris for visualization
            cv2.circle(frame, (int(iris_x), int(iris_y)), 3, (0, 255, 0), -1)

    cv2.imshow("Gaze Direction Tracking - Honest Gaze", frame)

    # Quit with "q"
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
