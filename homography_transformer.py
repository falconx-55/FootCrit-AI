import cv2
import numpy as np

class HomographyEngine:
    def __init__(self, src_pts=None, dst_pts=None):
        if src_pts is None or dst_pts is None:
            self.src_pts = np.float32([
                [400, 200],
                [1520, 200],
                [200, 900],
                [1720, 900]
            ])
            self.dst_pts = np.float32([
                [60, 0],
                [120, 0],
                [60, 80],
                [120, 80]
            ])
        else:
            self.src_pts = np.float32(src_pts)
            self.dst_pts = np.float32(dst_pts)

        self.h_matrix, _ = cv2.findHomography(self.src_pts, self.dst_pts)

    def transform_tracking_data(self, tracking_data):
        if not tracking_data:
            return []

        pixel_coords = np.array([[d['pixel_x'], d['pixel_y']] for d in tracking_data], dtype=np.float32)
        pixel_coords = np.array([pixel_coords])

        pitch_coords = cv2.perspectiveTransform(pixel_coords, self.h_matrix)
        pitch_coords = pitch_coords[0]

        transformed_data = []
        for i, data_point in enumerate(tracking_data):
            new_data_point = data_point.copy()
            new_data_point['pitch_x'] = float(pitch_coords[i][0])
            new_data_point['pitch_y'] = float(pitch_coords[i][1])
            transformed_data.append(new_data_point)

        return transformed_data