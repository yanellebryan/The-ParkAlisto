import cv2
import numpy as np

def load_parking_mask(mask_path, width, height):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Mask not found at {mask_path}, creating blank mask")
        return np.zeros((height, width), dtype=np.uint8)
    mask = cv2.resize(mask, (width, height))
    return mask

def get_parking_spots(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parking_spots = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 20 and h > 20:
            parking_spots.append((x, y, w, h))
    parking_spots.sort(key=lambda spot: spot[0])
    return parking_spots

def get_occupying_vehicle(spot_bbox, vehicles):
    spot_x, spot_y, spot_w, spot_h = spot_bbox
    spot_area = max(1, spot_w * spot_h)
    for vehicle in vehicles:
        veh_x1, veh_y1, veh_x2, veh_y2 = vehicle['bbox']
        inter_x1 = max(spot_x, veh_x1)
        inter_y1 = max(spot_y, veh_y1)
        inter_x2 = min(spot_x + spot_w, veh_x2)
        inter_y2 = min(spot_y + spot_h, veh_y2)
        
        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            overlap_ratio = intersection_area / spot_area
            if overlap_ratio > 0.3:
                return vehicle
    return None
