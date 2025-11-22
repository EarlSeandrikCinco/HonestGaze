"""
Project Honest Gaze — Full Prototype (Calibration + Micro-checks + Logging)

Features:
- Interactive 3x3 calibration (9 points)
- Linear mapping [h,v,1] -> screen_x, screen_y (least squares)
- Calibration quality check (normalized RMSE); auto-retry once if poor
- Save/load calibration coefficients to calib.npz
- Micro-checks during exam (randomized intervals) to validate ongoing calibration
- Long-gaze detection and burst detection using calibrated coordinates
- Event logging to events.csv for audit
- Manual controls: 'c' calibrate, 'r' reset calibration, 'q' quit
"""

import cv2
import time
import os
import csv
import math
import random
import numpy as np
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp

# -----------------------------
# Configurable parameters
# -----------------------------
SCREEN_W = 1280  # virtual screen size used for calibration mapping
SCREEN_H = 720
GRID = (3, 3)  # calibration grid cols x rows
SAMPLES_PER_POINT = 10  # frames captured per calibration target
HOLD_BEFORE = 0.6  # seconds to show dot before collecting
AUTO_RETRY = True  # allow one auto-retry on poor quality
RMSE_GOOD_THRESH = 0.06  # normalized RMSE (ratio of screen diagonal) considered good
RMSE_ACCEPT_THRESH = 0.12  # acceptable; above this is poor

# Proctor-configurable thresholds (can be exposed in a GUI later)
LONG_GAZE_THRESHOLD = 3  # seconds continuous looking away => long-gaze event
BURST_MAX_COUNT = 3  # short bursts allowed
BURST_WINDOW = 20  # seconds window for bursts
MICROCHECK_MIN = 30  # min seconds before first micro-check
MICROCHECK_MAX = 120  # max seconds between micro-checks
MICROCHECK_SAMPLES = 6  # number of frames collected for micro-check

# Output files
CALIB_FILE = "calib.npz"
EVENT_LOG = "events.csv"

# -----------------------------
# Mediapipe init
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
# Eye & indices (MediaPipe)
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

# Head pose points for optional use (not used in regression)
HEAD_POINTS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "mouth_left": 61,
    "mouth_right": 291
}

# Cooldown period after a warning before issuing another one (in seconds)
WARNING_COOLDOWN = 10

# -----------------------------
# Helper utilities
# -----------------------------
def log_event(event_type, details):
    """Append an event row to EVENT_LOG (timestamp,event_type,details)."""
    header_needed = not os.path.exists(EVENT_LOG)
    with open(EVENT_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if header_needed:
            writer.writerow(["timestamp", "event_type", "details"])
        writer.writerow([time.time(), event_type, details])


def safe_div(a, b):
    return a / (b + 1e-9)


# -----------------------------
# Feature extraction: hv ratio from landmarks (both eyes averaged)
# -----------------------------
def compute_hv(landmarks, w, h):
    """Return (h_ratio, v_ratio) where both eyes averaged.
       h_ratio: 0..1 (left->right), v_ratio: 0..1 (up->down)
    """
    # right eye
    rx = sum([landmarks[i].x for i in RIGHT_IRIS]) / len(RIGHT_IRIS) * w
    ry = sum([landmarks[i].y for i in RIGHT_IRIS]) / len(RIGHT_IRIS) * h
    r_left = landmarks[RIGHT_EYE_LEFT].x * w
    r_right = landmarks[RIGHT_EYE_RIGHT].x * w
    r_top = landmarks[RIGHT_EYE_TOP].y * h
    r_bottom = landmarks[RIGHT_EYE_BOTTOM].y * h
    hr_r = safe_div(rx - r_left, r_right - r_left)
    vr_r = 1.0 - safe_div(ry - r_top, r_bottom - r_top)  # invert so larger = looking down

    # left eye
    lx = sum([landmarks[i].x for i in LEFT_IRIS]) / len(LEFT_IRIS) * w
    ly = sum([landmarks[i].y for i in LEFT_IRIS]) / len(LEFT_IRIS) * h
    l_left = landmarks[LEFT_EYE_LEFT].x * w
    l_right = landmarks[LEFT_EYE_RIGHT].x * w
    l_top = landmarks[LEFT_EYE_TOP].y * h
    l_bottom = landmarks[LEFT_EYE_BOTTOM].y * h
    hr_l = safe_div(lx - l_left, l_right - l_left)
    vr_l = 1.0 - safe_div(ly - l_top, l_bottom - l_top)

    h_ratio = float(np.clip((hr_r + hr_l) / 2.0, 0.0, 1.0))
    v_ratio = float(np.clip((vr_r + vr_l) / 2.0, 0.0, 1.0))
    return h_ratio, v_ratio


# -----------------------------
# Calibration routine
# -----------------------------
def run_calibration(cap, face_mesh, screen_w, screen_h, grid=(3, 3),
                    samples_per_point=10, hold_before=0.6, abort_key='q'):
    """
    Run an interactive calibration; returns (coeffs_x, coeffs_y, rmse_norm)
    or (None, None, None) if aborted.
    """
    cols, rows = grid
    margin_x = 0.12
    margin_y = 0.12
    xs = np.linspace(margin_x * screen_w, (1 - margin_x) * screen_w, cols)
    ys = np.linspace(margin_y * screen_h, (1 - margin_y) * screen_h, rows)
    targets = [(int(x), int(y)) for y in ys for x in xs]  # row-major: top->bottom, left->right

    features = []
    screens = []

    cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("calib", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    instructions = "Calibration: look at the red dot. Press 'q' to abort."

    for (tx, ty) in targets:
        # show target briefly
        t0 = time.time()
        while time.time() - t0 < hold_before:
            blank = 255 * np.ones((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.putText(blank, instructions, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.circle(blank, (tx, ty), 18, (0, 0, 255), -1)
            cv2.imshow("calib", blank)
            if cv2.waitKey(1) & 0xFF == ord(abort_key):
                cv2.destroyWindow("calib")
                return None, None, None

        # collect frames for this target
        samples = []
        collected = 0
        while collected < samples_per_point:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            if not results.multi_face_landmarks:
                # show instruction and wait
                blank = 255 * np.ones((screen_h, screen_w, 3), dtype=np.uint8)
                cv2.putText(blank, "No face detected - align camera", (50, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.circle(blank, (tx, ty), 18, (0, 0, 255), -1)
                cv2.imshow("calib", blank)
                if cv2.waitKey(1) & 0xFF == ord(abort_key):
                    cv2.destroyWindow("calib")
                    return None, None, None
                continue

            face_landmarks = results.multi_face_landmarks[0]
            h_ratio, v_ratio = compute_hv(face_landmarks.landmark, frame.shape[1], frame.shape[0])
            samples.append((h_ratio, v_ratio))
            collected += 1

            # progress overlay
            blank = 255 * np.ones((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.circle(blank, (tx, ty), 18, (0, 0, 255), -1)
            cv2.putText(blank, f"Collecting {collected}/{samples_per_point}", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
            cv2.imshow("calib", blank)
            if cv2.waitKey(1) & 0xFF == ord(abort_key):
                cv2.destroyWindow("calib")
                return None, None, None

        # average samples
        avg_h = float(np.mean([s[0] for s in samples]))
        avg_v = float(np.mean([s[1] for s in samples]))
        features.append([avg_h, avg_v])
        screens.append([tx, ty])

    cv2.destroyWindow("calib")

    a = np.hstack([np.array(features), np.ones((len(features), 1))])  # Nx3
    screens = np.array(screens)  # Nx2

    coeffs_x, *_ = np.linalg.lstsq(a, screens[:, 0], rcond=None)
    coeffs_y, *_ = np.linalg.lstsq(a, screens[:, 1], rcond=None)

    # compute RMSE and normalized
    pred = a.dot(np.vstack([coeffs_x, coeffs_y]).T)
    residuals = np.linalg.norm(pred - screens, axis=1)
    rmse = math.sqrt(np.mean(residuals ** 2))
    diag = math.hypot(screen_w, screen_h)
    rmse_norm = rmse / diag

    np.savez(CALIB_FILE, coeffs_x=coeffs_x, coeffs_y=coeffs_y, grid=grid)
    return coeffs_x, coeffs_y, rmse_norm


# -----------------------------
# Prediction helper
# -----------------------------
def predict_screen_from_hv(h_ratio, v_ratio, coeffs_x, coeffs_y):
    a = np.array([h_ratio, v_ratio, 1.0])
    sx = float(a.dot(coeffs_x))
    sy = float(a.dot(coeffs_y))
    return sx, sy


# -----------------------------
# Micro-check routine (non-blocking)
# -----------------------------
def run_micro_check(cap, face_mesh, coeffs_x, coeffs_y, screen_w, screen_h,
                    samples=6, hold_before=0.2):
    """
    Shows a random dot briefly and checks predicted gaze proximity.
    Returns (pass_bool, predicted_center_dist_norm, dot_pos, details)
    """
    tx = random.randint(int(0.15 * screen_w), int(0.85 * screen_w))
    ty = random.randint(int(0.15 * screen_h), int(0.85 * screen_h))
    # show dot briefly (non-fullscreen small overlay)
    win = np.ones((screen_h, screen_w, 3), dtype=np.uint8) * 255
    cv2.circle(win, (tx, ty), 12, (0, 0, 255), -1)
    cv2.putText(win, "Micro-check: look at the dot", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.imshow("micro", win)
    t0 = time.time()
    while time.time() - t0 < hold_before:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow("micro")
            return False, None, (tx, ty), "aborted"

    collected = 0
    samples_list = []
    while collected < samples:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            continue
        face_landmarks = results.multi_face_landmarks[0]
        h_ratio, v_ratio = compute_hv(face_landmarks.landmark, frame.shape[1], frame.shape[0])
        samples_list.append((h_ratio, v_ratio))
        collected += 1
    cv2.destroyWindow("micro")

    if len(samples_list) == 0:
        return False, None, (tx, ty), "no_samples"

    avg_h = float(np.mean([s[0] for s in samples_list]))
    avg_v = float(np.mean([s[1] for s in samples_list]))

    sx, sy = predict_screen_from_hv(avg_h, avg_v, coeffs_x, coeffs_y)
    dx = sx - tx
    dy = sy - ty
    dist = math.hypot(dx, dy)
    diag = math.hypot(screen_w, screen_h)
    dist_norm = dist / diag

    details = f"pred=({sx:.1f},{sy:.1f}) dot=({tx},{ty}) dist_norm={dist_norm:.4f}"
    pass_check = dist_norm <= RMSE_ACCEPT_THRESH  # accept if within acceptable RMSE
    return pass_check, dist_norm, (tx, ty), details


# -----------------------------
# Main integrated loop
# -----------------------------
def main():
    # load calibration if exists else run
    coeffs_x = coeffs_y = None
    rmse_norm = None
    loaded = False

    if os.path.exists(CALIB_FILE):
        try:
            data = np.load(CALIB_FILE)
            coeffs_x = data["coeffs_x"]
            coeffs_y = data["coeffs_y"]
            print("Loaded existing calibration from", CALIB_FILE)
            loaded = True
        except Exception as e:
            print("Failed to load calibration:", e)

    if not loaded:
        # run calibration (auto-retry allowed)
        attempt = 0
        while True:
            attempt += 1
            print(f"Calibration attempt {attempt} ... (grid={GRID}, samples={SAMPLES_PER_POINT})")
            res = run_calibration(cap, face_mesh, SCREEN_W, SCREEN_H,
                                  grid=GRID, samples_per_point=SAMPLES_PER_POINT, hold_before=HOLD_BEFORE)
            if res[0] is None:
                print("Calibration aborted by user.")
                return
            coeffs_x, coeffs_y, rmse_norm = res
            print(f"Calibration RMSE (normalized): {rmse_norm:.4f}")
            if rmse_norm <= RMSE_GOOD_THRESH:
                print("Calibration quality: GOOD")
                log_event("calibration", f"good rmse={rmse_norm:.4f}")
                break
            elif rmse_norm <= RMSE_ACCEPT_THRESH:
                print("Calibration quality: ACCEPTABLE")
                log_event("calibration", f"acceptable rmse={rmse_norm:.4f}")
                break
            else:
                print("Calibration quality: POOR")
                log_event("calibration", f"poor rmse={rmse_norm:.4f}")
                if AUTO_RETRY and attempt == 1:
                    print("Auto-retrying calibration once due to poor quality...")
                    continue
                else:
                    # allow user to decide
                    msg = f"Calibration quality is poor (rmse={rmse_norm:.4f}). Re-run calibration?"
                    if messagebox.askyesno("Calibration poor", msg):
                        attempt = 0
                        continue
                    else:
                        print("Proceeding with poor calibration; results will be logged.")
                        break

    # set up micro-check timer
    next_micro = time.time() + random.uniform(MICROCHECK_MIN, MICROCHECK_MAX)
    last_warning_time = 0
    look_start = None
    burst_timestamps = []

    # open event log headers if needed
    if not os.path.exists(EVENT_LOG):
        log_event("session_start", f"coeff_loaded={bool(coeffs_x is not None)} rmse={rmse_norm}")

    print("Starting main loop. Controls: 'c' calibrate, 'r' reset, 'q' quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam error.")
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape
        current_time = time.time()

        # top-level keyboard handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit requested.")
            break
        if key == ord('c'):
            # re-run calibration manually
            res = run_calibration(cap, face_mesh, SCREEN_W, SCREEN_H,
                                  grid=GRID, samples_per_point=SAMPLES_PER_POINT, hold_before=HOLD_BEFORE)
            if res[0] is not None:
                coeffs_x, coeffs_y, rmse_norm = res
                log_event("calibration", f"manual rmse={rmse_norm:.4f}")
                print("Calibration updated (manual). rmse:", rmse_norm)
            continue
        if key == ord('r'):
            # delete saved calibration
            try:
                if os.path.exists(CALIB_FILE):
                    os.remove(CALIB_FILE)
                coeffs_x = coeffs_y = None
                print("Calibration reset. Re-run calibration now.")
                log_event("calibration_reset", "user requested")
                # force calibration loop
                res = run_calibration(cap, face_mesh, SCREEN_W, SCREEN_H,
                                      grid=GRID, samples_per_point=SAMPLES_PER_POINT, hold_before=HOLD_BEFORE)
                if res[0] is not None:
                    coeffs_x, coeffs_y, rmse_norm = res
                    log_event("calibration", f"after reset rmse={rmse_norm:.4f}")
                else:
                    print("Calibration aborted after reset.")
                    return
            except Exception as e:
                print("Reset failed:", e)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h_ratio, v_ratio = compute_hv(face_landmarks.landmark, w, h)

            # predict screen coords using calibration mapping
            if coeffs_x is not None:
                sx, sy = predict_screen_from_hv(h_ratio, v_ratio, coeffs_x, coeffs_y)
                # compute normalized distance to center
                cx, cy = SCREEN_W / 2.0, SCREEN_H / 2.0
                dx = sx - cx
                dy = sy - cy
                dist = math.hypot(dx, dy)
                diag = math.hypot(SCREEN_W, SCREEN_H)
                dist_norm = dist / diag
                # decide looking_away using threshold proportional to calibration rmse
                # dynamic threshold: base 0.12 plus rmse_norm * 2 (tunable)
                base_threshold = 0.12
                dynamic_thresh = base_threshold + (rmse_norm if rmse_norm is not None else 0.06) * 2.0
                looking_away = dist_norm > dynamic_thresh
                gaze_label = "CENTER" if not looking_away else "AWAY"
            else:
                # fallback to raw horizontal thresholds if no calib
                gaze_label = "CENTER"
                looking_away = False
                sx = sy = dist_norm = None

            # long gaze detection logic
            if looking_away:
                if look_start is None:
                    look_start = current_time
                else:
                    if current_time - look_start >= LONG_GAZE_THRESHOLD:
                        if current_time - last_warning_time >= WARNING_COOLDOWN:
                            message = f"Long gaze detected dist_norm={dist_norm:.4f} thresh={dynamic_thresh:.4f}"
                            print("[WARNING]", message)
                            messagebox.showwarning("Long Gaze Warning", "You have looked away for too long.")
                            log_event("long_gaze", message)
                            last_warning_time = current_time
                            look_start = None
            else:
                look_start = None

            # burst detection
            if looking_away:
                burst_timestamps.append(current_time)
            burst_timestamps = [t for t in burst_timestamps if current_time - t <= BURST_WINDOW]
            if len(burst_timestamps) >= BURST_MAX_COUNT:
                if current_time - last_warning_time >= WARNING_COOLDOWN:
                    message = f"Burst gaze detected count={len(burst_timestamps)} window={BURST_WINDOW}"
                    print("[WARNING]", message)
                    messagebox.showwarning("Burst Gaze Warning", "Repeated short glances detected.")
                    log_event("burst_gaze", message)
                    last_warning_time = current_time
                    burst_timestamps.clear()

            # display overlay
            overlay = frame.copy()
            txt = f"Gaze:{gaze_label} dist_norm:{dist_norm:.3f}" if dist_norm is not None else f"Gaze:{gaze_label}"
            cv2.putText(overlay, txt, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if sx is not None:
                cv2.putText(overlay, f"pred:({int(sx)},{int(sy)})", (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 1)
            if rmse_norm is not None:
                cv2.putText(overlay, f"calib_rmse:{rmse_norm:.4f}", (14, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 1)
            cv2.circle(overlay, (int(h_ratio * w), int((1 - v_ratio) * h)), 3, (0, 255, 0), -1)  # eye marker
            frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        # show frame
        cv2.imshow("Honest Gaze - Proctoring", frame)

        # handle micro-check scheduling
        if coeffs_x is not None and time.time() >= next_micro:
            # run micro-check asynchronously (blocking minimal time)
            pass_check, dist_norm_micro, dot, details = run_micro_check(cap, face_mesh, coeffs_x, coeffs_y, SCREEN_W,
                                                                        SCREEN_H, samples=MICROCHECK_SAMPLES)
            if not pass_check:
                print("[MICROCHECK FAIL]", details)
                log_event("microcheck_fail", details)
                # optional: escalate as warning
                if time.time() - last_warning_time >= WARNING_COOLDOWN:
                    messagebox.showwarning("Micro-check failed", "Gaze did not match micro-check dot.")
                    last_warning_time = time.time()
            else:
                print("[MICROCHECK PASS]", details)
                log_event("microcheck_pass", details)
            # schedule next micro-check
            next_micro = time.time() + random.uniform(MICROCHECK_MIN, MICROCHECK_MAX)

    # end loop
    cap.release()
    cv2.destroyAllWindows()
    log_event("session_end", "user_quit")


if __name__ == "__main__":
    main()
