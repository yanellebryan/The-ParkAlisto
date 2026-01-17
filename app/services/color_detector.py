import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

class ColorDetector:
    def __init__(self):
        self.colors = {
            'White': (240, 240, 240), 
            'Black': (30, 30, 30), 
            'Red': (50, 50, 180),
            'Blue': (180, 100, 50), 
            'Green': (50, 150, 50), 
            'Yellow': (50, 200, 200),
            'Gray': (128, 128, 128),
            'Silver': (192, 192, 192)
        }

    def get_vehicle_color(self, image_roi):
        if image_roi is None or image_roi.size == 0:
            return "N/A"
        try:
            resized = cv2.resize(image_roi, (50, 50))
            pixels = resized.reshape(-1, 3).astype(np.float32)
            
            kmeans = MiniBatchKMeans(n_clusters=1, random_state=0, n_init=10).fit(pixels)
            dominant_bgr = kmeans.cluster_centers_[0].astype(int)
            
            min_dist = float('inf')
            closest_color = "Unknown"
            for name, bgr in self.colors.items():
                dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(dominant_bgr, bgr)))
                if dist < min_dist:
                    min_dist = dist
                    closest_color = name
            return closest_color
        except Exception as e:
            print(f"Color detection error: {e}")
            return "N/A"

color_detector = ColorDetector()
