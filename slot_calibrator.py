import cv2
import json

IMAGE = "calibration.jpg"

img = cv2.imread(IMAGE)
display = img.copy()

slots = {}
current_points = []
slot_id = 0


def mouse(event, x, y, flags, param):
    global current_points, display

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append((x, y))
        cv2.circle(display, (x, y), 4, (0,255,0), -1)


cv2.namedWindow("calib")
cv2.setMouseCallback("calib", mouse)

print("Instructions:")
print("Click polygon corners for each slot.")
print("Press N = next slot")
print("Press U = undo last point")
print("Press S = save")
print("Press Q = quit")

while True:
    cv2.imshow("calib", display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("n"):
        if len(current_points) >= 3:
            slots[slot_id] = current_points.copy()
            print(f"Saved slot {slot_id} with {len(current_points)} points")
            slot_id += 1
            current_points = []
        else:
            print("Need at least 3 points")

    elif key == ord("u"):
        if current_points:
            current_points.pop()
            display = img.copy()
            for pts in slots.values():
                for p in pts:
                    cv2.circle(display, p, 4, (255,0,0), -1)
            for p in current_points:
                cv2.circle(display, p, 4, (0,255,0), -1)

    elif key == ord("s"):
        with open("slots.json", "w") as f:
            json.dump(slots, f)
        print("Saved slots.json")

    elif key == ord("q"):
        break

cv2.destroyAllWindows()
