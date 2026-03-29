import cv2
import numpy as np

class RampCalibrator:
    def __init__(self):
        self.image_points = []
        self.homography = None

    def add_point(self, x, y):
        self.image_points.append([x, y])

    def compute(self, ramp_width=900, ramp_height=300):
        if len(self.image_points) != 4:
            raise ValueError("Need 4 points")

        src = np.array(self.image_points, dtype=np.float32)
        dst = np.array([
            [0, 0],
            [ramp_width, 0],
            [ramp_width, ramp_height],
            [0, ramp_height]
        ], dtype=np.float32)

        self.homography, _ = cv2.findHomography(src, dst)

    def pixel_to_ramp(self, x, y):
        if self.homography is None:
            return None

        point = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, self.homography)
        return mapped[0][0]
