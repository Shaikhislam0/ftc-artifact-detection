import cv2

VIDEO = "20260202_161511.mp4"

cap = cv2.VideoCapture(VIDEO)
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

ret, frame = cap.read()

if ret:
    cv2.imwrite("calibration.jpg", frame)
    print("Saved calibration.jpg")

cap.release()
