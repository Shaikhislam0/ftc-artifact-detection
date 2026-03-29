"""
FTC Artifact Detection System
==============================
Pure color-based detection pipeline for green and purple artifacts.
No YOLO — uses HSV masking + contour detection.
Auto-calibrates HSV ranges on first N frames to handle varying lighting.

Author: Shaikhislam Askanbay
"""

import cv2
import numpy as np
from collections import deque


# ── Configuration ─────────────────────────────────────────────────────────────

VIDEO_SOURCE   = 0           # 0 = webcam, or "video.mp4" for file
CALIBRATE_FRAMES = 30        # frames to sample for auto HSV calibration
MIN_RADIUS     = 12          # minimum blob radius to count as artifact (px)
MAX_RADIUS     = 80          # maximum blob radius
STABLE_FRAMES  = 8           # frames blob must persist to be "confirmed"
SLOT_PROXIMITY = 60          # px distance to snap blob to nearest slot

# Fallback HSV ranges (used if calibration fails)
GREEN_HSV_LOW  = np.array([35,  60,  60])
GREEN_HSV_HIGH = np.array([85, 255, 255])
PURP_HSV_LOW   = np.array([120, 50,  50])
PURP_HSV_HIGH  = np.array([170, 255, 255])


# ── Auto HSV Calibrator ────────────────────────────────────────────────────────

class HSVCalibrator:
    """
    Samples the first N frames and auto-adjusts HSV ranges
    by analyzing the dominant color clusters in the scene.
    Handles lighting variance by widening tolerances adaptively.
    """
    def __init__(self, n_frames=CALIBRATE_FRAMES):
        self.n       = n_frames
        self.count   = 0
        self.done    = False
        self.g_samples = []
        self.p_samples = []

        # Start with fallback ranges
        self.green_low  = GREEN_HSV_LOW.copy()
        self.green_high = GREEN_HSV_HIGH.copy()
        self.purp_low   = PURP_HSV_LOW.copy()
        self.purp_high  = PURP_HSV_HIGH.copy()

    def feed(self, frame):
        if self.done:
            return
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Sample green pixels using loose range
        g_mask = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)
        g_mask = cv2.erode(g_mask, None, iterations=2)
        g_pixels = hsv[g_mask > 0]
        if len(g_pixels) > 20:
            self.g_samples.append(g_pixels)

        # Sample purple pixels using loose range
        p_mask = cv2.inRange(hsv, PURP_HSV_LOW, PURP_HSV_HIGH)
        p_mask = cv2.erode(p_mask, None, iterations=2)
        p_pixels = hsv[p_mask > 0]
        if len(p_pixels) > 20:
            self.p_samples.append(p_pixels)

        self.count += 1
        if self.count >= self.n:
            self._compute()
            self.done = True
            print("[Calibration complete]")
            print(f"  Green HSV: {self.green_low} → {self.green_high}")
            print(f"  Purple HSV: {self.purp_low} → {self.purp_high}")

    def _compute(self):
        MARGIN = np.array([8, 40, 40])  # tolerance margin per channel

        if self.g_samples:
            all_g = np.vstack(self.g_samples)
            self.green_low  = np.clip(all_g.min(axis=0) - MARGIN, 0, 255)
            self.green_high = np.clip(all_g.max(axis=0) + MARGIN, 0, 255)

        if self.p_samples:
            all_p = np.vstack(self.p_samples)
            self.purp_low  = np.clip(all_p.min(axis=0) - MARGIN, 0, 255)
            self.purp_high = np.clip(all_p.max(axis=0) + MARGIN, 0, 255)


# ── Blob Detector ──────────────────────────────────────────────────────────────

def detect_blobs(frame, hsv, low, high, label, color_bgr):
    """
    Detect circular blobs of a given HSV range.
    Returns list of (cx, cy, radius, label).
    """
    mask = cv2.inRange(hsv, low, high)

    # Clean noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < np.pi * MIN_RADIUS**2:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cx, cy, radius = int(cx), int(cy), int(radius)

        if not (MIN_RADIUS <= radius <= MAX_RADIUS):
            continue

        # Circularity check — artifacts are round
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.55:
            continue

        detections.append((cx, cy, radius, label, color_bgr))

    return detections


# ── Slot Manager ──────────────────────────────────────────────────────────────

class SlotManager:
    """
    Tracks artifact positions relative to named slots.
    Slots are defined by clicking on the first frame (optional).
    If no slots defined, just tracks raw positions.
    """
    def __init__(self):
        self.slots   = {}    # slot_id -> (x, y)
        self.state   = {}    # slot_id -> label
        self.history = {}    # slot_id -> deque of recent labels

    def define_slot(self, slot_id, x, y):
        self.slots[slot_id]   = (x, y)
        self.state[slot_id]   = "NONE"
        self.history[slot_id] = deque(maxlen=STABLE_FRAMES)

    def update(self, detections):
        # Reset all slots
        for sid in self.slots:
            self.history[sid].append("NONE")

        for (cx, cy, r, label, _) in detections:
            best_sid  = None
            best_dist = SLOT_PROXIMITY

            for sid, (sx, sy) in self.slots.items():
                dist = np.hypot(cx - sx, cy - sy)
                if dist < best_dist:
                    best_dist = dist
                    best_sid  = sid

            if best_sid is not None:
                self.history[best_sid][-1] = label

        # Confirm stable readings
        for sid in self.slots:
            hist = list(self.history[sid])
            if len(hist) >= STABLE_FRAMES:
                majority = max(set(hist), key=hist.count)
                self.state[sid] = majority

    def get_state(self):
        return dict(self.state)


# ── Slot Click Setup ──────────────────────────────────────────────────────────

slot_manager  = SlotManager()
slot_counter  = [0]
setup_mode    = [True]

def on_mouse(event, x, y, flags, param):
    if not setup_mode[0]:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        sid = slot_counter[0]
        slot_manager.define_slot(sid, x, y)
        print(f"[Setup] Slot {sid} defined at ({x}, {y})")
        slot_counter[0] += 1


# ── Main Loop ─────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print("ERROR: Cannot open video source.")
        return

    calibrator = HSVCalibrator()

    cv2.namedWindow("FTC Artifact Detection")
    cv2.setMouseCallback("FTC Artifact Detection", on_mouse)

    print("=" * 50)
    print("FTC Artifact Detection System")
    print("SETUP MODE: Click to define slots on the frame.")
    print("Press SPACE to finish setup and start detection.")
    print("Press R to recalibrate HSV.")
    print("Press ESC to quit.")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        display = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Feed calibrator
        if not calibrator.done:
            calibrator.feed(frame)
            cv2.putText(display,
                f"Calibrating... {calibrator.count}/{calibrator.n}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        # ── Setup mode: show slot markers ────────────────────────────────────
        if setup_mode[0]:
            for sid, (sx, sy) in slot_manager.slots.items():
                cv2.circle(display, (sx, sy), 8, (255, 255, 0), -1)
                cv2.putText(display, f"S{sid}", (sx + 10, sy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(display, "SETUP: click slots | SPACE to start",
                        (10, display.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # ── Detection mode ────────────────────────────────────────────────────
        else:
            green_dets  = detect_blobs(frame, hsv,
                                       calibrator.green_low, calibrator.green_high,
                                       "G", (0, 200, 0))
            purple_dets = detect_blobs(frame, hsv,
                                       calibrator.purp_low, calibrator.purp_high,
                                       "P", (180, 0, 200))
            all_dets = green_dets + purple_dets

            # Draw detections
            for (cx, cy, radius, label, color_bgr) in all_dets:
                cv2.circle(display, (cx, cy), radius, color_bgr, 2)
                cv2.circle(display, (cx, cy), 4, (255, 255, 255), -1)
                cv2.putText(display, label, (cx - 8, cy - radius - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_bgr, 2)

            # Update slots
            if slot_manager.slots:
                slot_manager.update(all_dets)
                state = slot_manager.get_state()

                # Draw slot states
                for sid, (sx, sy) in slot_manager.slots.items():
                    label = state.get(sid, "NONE")
                    col = (0, 200, 0) if label == "G" else \
                          (180, 0, 200) if label == "P" else (80, 80, 80)
                    cv2.circle(display, (sx, sy), 18, col, 2)
                    cv2.putText(display, f"S{sid}:{label}",
                                (sx - 20, sy + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            # Detection count
            g_count = sum(1 for d in all_dets if d[3] == "G")
            p_count = sum(1 for d in all_dets if d[3] == "P")
            cv2.putText(display, f"G:{g_count}  P:{p_count}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("FTC Artifact Detection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            break
        elif key == 32: # SPACE — end setup
            setup_mode[0] = False
            print(f"[Detection started] {len(slot_manager.slots)} slots defined.")
        elif key == ord('r'):
            calibrator.done  = False
            calibrator.count = 0
            calibrator.g_samples.clear()
            calibrator.p_samples.clear()
            print("[Recalibrating...]")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
