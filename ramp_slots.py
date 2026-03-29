
import cv2
import numpy as np

class RampSlots:
    def __init__(self):
        self.slots = {}

    def add_slot(self, slot_id, points):
        self.slots[slot_id] = np.array(points, dtype=np.int32)

    def get_slot(self, x, y):
        for slot_id, poly in self.slots.items():
            inside = cv2.pointPolygonTest(poly, (x, y), False)
            if inside >= 0:
                return slot_id
        return None
