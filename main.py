import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------
# Load YOLO model
# -------------------------------
model = YOLO("yolov8n.pt")  # temporary model for testing


# -------------------------------
# Color detection function
# -------------------------------
def classify_color(image_crop):

    hsv = cv2.cvtColor(image_crop, cv2.COLOR_BGR2HSV)

    # Green HSV range
    green_low = np.array([40, 50, 50])
    green_high = np.array([80, 255, 255])

    # Purple HSV range
    purple_low = np.array([125, 50, 50])
    purple_high = np.array([165, 255, 255])

    green_mask = cv2.inRange(hsv, green_low, green_high)
    purple_mask = cv2.inRange(hsv, purple_low, purple_high)

    green_pixels = cv2.countNonZero(green_mask)
    purple_pixels = cv2.countNonZero(purple_mask)

    if green_pixels > purple_pixels:
        return "G"
    elif purple_pixels > green_pixels:
        return "P"
    else:
        return "NONE"


# -------------------------------
# Video Input
# -------------------------------
cap = cv2.VideoCapture("video.mp4")  # change to webcam later if needed

while True:

    ret, frame = cap.read()
    if not ret:
        break
        
    # Run YOLO detection
    results = model(frame)

    for r in results:

        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            color = classify_color(crop)

            # Artifact center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw results
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)

            cv2.putText(frame, color, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("FTC Artifact Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
