import torch
import cv2
from app.core.constants import VEHICLE_CLASSES, VEHICLE_NAMES

class AIDetector:
    def __init__(self):
        self.model = None
        
    def load_model(self):
        print("Loading YOLOv5 Model...")
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
            self.model.conf = 0.25
            self.model.iou = 0.45
            print("✓ YOLOv5 loaded successfully")
        except Exception as e:
            print(f"Error loading YOLOv5: {e}")
    
    def detect(self, frame):
        if self.model is None:
            self.load_model()
            if self.model is None:
                return []
                
        results = self.model(frame)
        vehicle_info_list = []
        
        try:
            det = results.pandas().xyxy[0]
            # Assuming frame is already resized to PROCESSING_WIDTH/HEIGHT passed to detect?
            # Or detect takes full frame?
            # In parkalisto.py, frame is resized before passing to model.
            # But here let's assume we pass the frame as is and get results relative to it?
            # Wait, parkalisto.py logic:
            # processed_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
            # results = model(processed_frame)
            # scale_x = width / PROCESSING_WIDTH ...
            
            # Use raw results here, scaling handled by caller or here if we pass orig dims.
            # Let's return the simplified detection data.
            for _, d in det.iterrows():
                if int(d['class']) in VEHICLE_CLASSES:
                    vehicle_info_list.append({
                        'xmin': int(d['xmin']),
                        'ymin': int(d['ymin']),
                        'xmax': int(d['xmax']),
                        'ymax': int(d['ymax']),
                        'type': VEHICLE_NAMES.get(int(d['class']), 'vehicle'),
                        'confidence': float(d['confidence'])
                    })
        except Exception as e:
            print(f"YOLO Detection Error: {e}")
            
        return vehicle_info_list

vehicle_detector = AIDetector()
