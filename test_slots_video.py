import cv2
from slot_lookup import SlotLookup

VIDEO = "20260202_161511.mp4"

lookup = SlotLookup()

cap = cv2.VideoCapture(VIDEO)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    for sid, poly in lookup.slots.items():
        cv2.polylines(frame, [poly], True, (0,255,255), 2)

        cx = poly[:,0].mean().astype(int)
        cy = poly[:,1].mean().astype(int)

        cv2.putText(frame, str(sid),
                    (cx,cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255),
                    2)

    cv2.imshow("slots", frame)

    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()
