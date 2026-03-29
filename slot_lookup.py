import json
import cv2
import numpy as np


class SlotLookup:

    def __init__(self, path="slots.json"):
        with open(path) as f:
            raw = json.load(f)

        self.slots = {
            int(k): np.array(v, dtype=np.int32)
            for k, v in raw.items()
        }

    def find_slot(self, x, y):
        for sid, poly in self.slots.items():
            if cv2.pointPolygonTest(poly, (x,y), False) >= 0:
                return sid
        return None
