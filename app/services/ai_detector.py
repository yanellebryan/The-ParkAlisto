import torch
import cv2
from app.core.constants import VEHICLE_CLASSES, VEHICLE_NAMES

class AIDetector:
    def __init__(self):
        self.model = None
        
    def load_model(self):
        print("Loading YOLOv5 Model (Local Cache)...")
        try:
            import os
            # Use the cached ultralytics_yolov5_master repo found in ~/.cache/torch/hub
            # This avoids the 'ultralytics' package trying to download/update the model weights
            hub_path = os.path.expanduser('~/.cache/torch/hub/ultralytics_yolov5_master')
            
            if not os.path.exists(hub_path):
                 print(f"Error: Cached YOLOv5 repo not found at {hub_path}")
                 return

            # Load using torch.hub from local source
            self.model = torch.hub.load(hub_path, 'custom', path='yolov5s.pt', source='local')
            
            # Configure model settings
            self.model.conf = 0.25
            self.model.iou = 0.45
            print(f"✓ YOLOv5 loaded from local cache")
        except Exception as e:
            print(f"Error loading YOLOv5: {e}")
    
    def detect(self, frame):
        if self.model is None:
            self.load_model()
            if self.model is None:
                return []
                
        # Inference using the torch hub model (pandas results)
        results = self.model(frame)
        vehicle_info_list = []
        
        try:
            # The torch hub model returns a Detections object where .pandas().xyxy[0] works
            det = results.pandas().xyxy[0]
            
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
