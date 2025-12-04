# This script runs a Flask web server to provide real-time
# parking lot data and video streams from MULTIPLE parking lots.
# NOW WITH MULTI-VIDEO SUPPORT AND BACKGROUND PROCESSING!

import os
import time
import threading
from datetime import datetime
from collections import defaultdict, deque
import json
import cv2
import numpy as np
import torch
import easyocr
from flask import Flask, Response, jsonify, send_from_directory
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import sys
import traceback
from queue import Queue

app = Flask(__name__)

# ==============================
# CONFIG
# ==============================
# Confirmation timers (seconds)
OCCUPANCY_CONFIRMATION_TIME = 10.0
VACANCY_CONFIRMATION_TIME = 10.0
SCREENSHOT_DELAY_AFTER_CONFIRMATION = 5.0

# Performance settings
FRAME_SKIP_RATE = 10
PROCESSING_WIDTH = 416
PROCESSING_HEIGHT = 416

# Background processing settings
NUM_PROCESSING_THREADS = 3

# Create directories for storing plate screenshots
PLATE_SCREENSHOTS_DIR = "plate_screenshots"
if not os.path.exists(PLATE_SCREENSHOTS_DIR):
    os.makedirs(PLATE_SCREENSHOTS_DIR)

# Vehicle classes from COCO
VEHICLE_CLASSES = [2, 3, 5, 7]
VEHICLE_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# ==============================
# Multi-Video Configuration
# ==============================
class VideoConfig:
    def __init__(self, lot_id, lot_name, video_path, mask_path):
        self.lot_id = lot_id
        self.lot_name = lot_name
        self.video_path = video_path
        self.mask_path = mask_path
        self.frame = None
        self.data = {
            'lot_id': lot_id,
            'lot_name': lot_name,
            'occupied': 0,
            'total': 0,
            'pending': 0,
            'last_update': 'N/A',
            'queue_size': 0
        }
        self.history = deque(maxlen=20)
        self.active_log = {}
        self.frame_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.log_lock = threading.Lock()

# Global list of video configurations
video_configs = []
combined_frame = None
combined_frame_lock = threading.Lock()

# Shared processing queue for all videos
processing_queue = Queue()

# ==============================
# YOLOv5 model (COCO)
# ==============================
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.conf = 0.25
model.iou = 0.45

# ==============================
# EasyOCR for license plate recognition
# ==============================
plate_reader = easyocr.Reader(['en'])

# ==============================
# Background Processing Task Queue
# ==============================
class ProcessingTask:
    def __init__(self, lot_id, spot_id, vehicle_roi, vehicle_type, timestamp):
        self.lot_id = lot_id
        self.spot_id = spot_id
        self.vehicle_roi = vehicle_roi
        self.vehicle_type = vehicle_type
        self.timestamp = timestamp
        self.screenshot_filename = None

def background_processor_worker():
    """Worker thread that processes plate and color detection in the background."""
    print(f"[WORKER] Background processor thread started (ID: {threading.get_ident()})")
    
    while True:
        try:
            task = processing_queue.get()
            
            if task is None:
                print(f"[WORKER] Received stop signal")
                break
            
            print(f"[WORKER] Processing lot {task.lot_id}, spot {task.spot_id} (queue size: {processing_queue.qsize()})")
            
            # Save screenshot
            screenshot_filename = f"lot{task.lot_id}_spot{task.spot_id}_{int(time.time())}.jpg"
            screenshot_path = os.path.join(PLATE_SCREENSHOTS_DIR, screenshot_filename)
            try:
                cv2.imwrite(screenshot_path, task.vehicle_roi)
                task.screenshot_filename = screenshot_filename
                print(f"[WORKER] Saved screenshot: {screenshot_filename}")
            except Exception as e:
                print(f"[WORKER] Error saving screenshot: {e}")
            
            start_time = time.time()
            
            # Detect plate number
            plate_number = recognize_plate(task.vehicle_roi)
            plate_time = time.time() - start_time
            print(f"[WORKER] Plate detection took {plate_time:.2f}s: {plate_number}")
            
            # Detect color
            color_start = time.time()
            color = get_vehicle_color(task.vehicle_roi)
            color_time = time.time() - color_start
            print(f"[WORKER] Color detection took {color_time:.2f}s: {color}")
            
            total_time = time.time() - start_time
            print(f"[WORKER] Total processing time: {total_time:.2f}s")
            
            # Update the parking log for the specific video
            config = video_configs[task.lot_id]
            update_processing_results(config, task.spot_id, plate_number, color, task.screenshot_filename)
            
            processing_queue.task_done()
            
        except Exception as e:
            print(f"[WORKER] Error in background processor: {e}")
            traceback.print_exc()
            processing_queue.task_done()

# ==============================
# Parking History Log Functions
# ==============================
def log_car_parked(config, spot_id, vehicle_type, vehicle_roi=None):
    """Logs a new car parking event - screenshot will be taken after delay"""
    timestamp = datetime.now().strftime("%I:%M:%S %p %B %d, %Y")
    
    with config.log_lock:
        log_entry = {
            'lot_id': config.lot_id,
            'lot_name': config.lot_name,
            'spot_id': spot_id,
            'plate_number': 'Waiting for screenshot...',  # CHANGED
            'color': 'Waiting for screenshot...',  # CHANGED
            'vehicle_type': vehicle_type,
            'timestamp_in': timestamp,
            'is_active': True,
            'plate_image': None,
            'processing_status': 'waiting_for_screenshot'  # CHANGED
        }
        config.active_log[spot_id] = log_entry
        config.history.appendleft(log_entry.copy())
        print(f"LOG: Car ({vehicle_type}) confirmed at lot {config.lot_id}, spot {spot_id}. Waiting {SCREENSHOT_DELAY_AFTER_CONFIRMATION}s before screenshot...")  # CHANGED
    
    if vehicle_roi is not None:
        task = ProcessingTask(config.lot_id, spot_id, vehicle_roi.copy(), vehicle_type, timestamp)
        processing_queue.put(task)
        print(f"LOG: Queued processing task for lot {config.lot_id}, spot {spot_id} (queue size: {processing_queue.qsize()})")


def queue_screenshot_processing(config, spot_id, vehicle_type, vehicle_roi):
    """Queue the screenshot and processing after the delay has elapsed"""
    timestamp = datetime.now().strftime("%I:%M:%S %p %B %d, %Y")
    
    with config.log_lock:
        if spot_id in config.active_log:
            config.active_log[spot_id]['processing_status'] = 'queued'
            config.active_log[spot_id]['plate_number'] = 'Processing...'
            config.active_log[spot_id]['color'] = 'Processing...'
            
            for entry in config.history:
                if (entry['spot_id'] == spot_id and 
                    entry['is_active'] and
                    entry['lot_id'] == config.lot_id):
                    entry['processing_status'] = 'queued'
                    entry['plate_number'] = 'Processing...'
                    entry['color'] = 'Processing...'
                    break
    
    task = ProcessingTask(config.lot_id, spot_id, vehicle_roi.copy(), vehicle_type, timestamp)
    processing_queue.put(task)
    print(f"LOG: Screenshot taken for lot {config.lot_id}, spot {spot_id}. Queued for processing (queue size: {processing_queue.qsize()})")    


def update_processing_results(config, spot_id, plate_number, color, screenshot_filename):
    """Updates a log entry with the results from background processing."""
    with config.log_lock:
        if spot_id in config.active_log:
            log_entry = config.active_log[spot_id]
            log_entry['plate_number'] = plate_number
            log_entry['color'] = color
            log_entry['plate_image'] = screenshot_filename
            log_entry['processing_status'] = 'completed'
            
            for entry in config.history:
                if (entry['spot_id'] == spot_id and 
                    entry['timestamp_in'] == log_entry['timestamp_in'] and
                    entry['is_active'] and
                    entry['lot_id'] == config.lot_id):
                    entry['plate_number'] = plate_number
                    entry['color'] = color
                    entry['plate_image'] = screenshot_filename
                    entry['processing_status'] = 'completed'
                    break
            
            print(f"LOG: Updated lot {config.lot_id}, spot {spot_id} - Plate: {plate_number}, Color: {color}")
        else:
            print(f"LOG: Warning - lot {config.lot_id}, spot {spot_id} not found in active log")

def log_car_left(config, spot_id):
    """Updates a log entry when a car leaves."""
    with config.log_lock:
        if spot_id in config.active_log:
            log_entry = config.active_log[spot_id]
            log_entry['timestamp_out'] = datetime.now().strftime("%I:%M:%S %p %B %d, %Y")
            log_entry['is_active'] = False
            del config.active_log[spot_id]
            print(f"LOG: Car from lot {config.lot_id}, spot {spot_id} has left.")

# ==============================
# Utility Functionslog_car_parked
# ==============================
def load_parking_mask(mask_path, width, height):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Mask image not found or invalid: {mask_path}")
    mask = cv2.resize(mask, (width, height))
    return mask

def get_parking_spots(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parking_spots = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
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

def get_vehicle_color(image_roi):
    """Analyzes the dominant color in a cropped image region."""
    if image_roi is None or image_roi.size == 0:
        return "N/A"
    
    try:
        resized = cv2.resize(image_roi, (50, 50))
        pixels = resized.reshape(-1, 3).astype(np.float32)
        
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=1, random_state=0, n_init=10).fit(pixels)
        dominant_bgr = kmeans.cluster_centers_[0].astype(int)
        dominant_bgr = tuple(map(int, dominant_bgr))
        
        colors = {
            'white': (240, 240, 240),
            'black': (30, 30, 30),
            'gray': (128, 128, 128),
            'silver': (192, 192, 192),
            'red': (50, 50, 180),
            'dark red': (30, 30, 120),
            'green': (50, 150, 50),
            'dark green': (30, 90, 30),
            'blue': (180, 100, 50),
            'dark blue': (120, 60, 30),
            'yellow': (50, 200, 200),
            'cyan': (200, 200, 50),
            'magenta': (180, 50, 180),
            'orange': (50, 140, 220),
            'brown': (42, 42, 120),
            'beige': (150, 180, 200)
        }
        
        min_dist = float('inf')
        closest_color_name = "unknown"
        
        for name, bgr in colors.items():
            dist = np.sqrt(sum((a - b) ** 2 for a, b in zip(dominant_bgr, bgr)))
            if dist < min_dist:
                min_dist = dist
                closest_color_name = name
        
        return closest_color_name
        
    except Exception as e:
        print(f"[COLOR] Detection error: {e}")
        return "N/A"

def preprocess_for_ocr(image):
    """Enhanced preprocessing for better OCR results on license plates."""
    if image is None or image.size == 0:
        return None
    
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    scale_factor = max(3.0, 80/h, 300/w)
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    processed_versions = []
    
    _, thresh1 = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_versions.append(thresh1)
    
    adaptive = cv2.adaptiveThreshold(
        resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 
        max(3, int(new_h/10)), 2
    )
    processed_versions.append(adaptive)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(resized)
    _, thresh2 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_versions.append(thresh2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    morph = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    processed_versions.append(morph)
    
    best_version = processed_versions[0]
    best_score = 0
    
    for version in processed_versions:
        hist = cv2.calcHist([version], [0], None, [256], [0, 256])
        black_pixels = hist[0]
        white_pixels = hist[255]
        total_pixels = version.shape[0] * version.shape[1]
        
        if total_pixels > 0:
            black_ratio = black_pixels / total_pixels
            white_ratio = white_pixels / total_pixels
            score = min(black_ratio, white_ratio) * (1 - abs(black_ratio - white_ratio))
            if score > best_score:
                best_score = score
                best_version = version
    
    return best_version

def find_plate_regions(image_roi):
    """Try to find rectangular regions that might contain license plates."""
    if image_roi is None or image_roi.size == 0:
        return [image_roi]
    
    if len(image_roi.shape) == 3:
        gray = cv2.cvtColor(image_roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_roi.copy()
    
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plate_candidates = []
    h, w = gray.shape
    
    for contour in contours:
        x, y, rect_w, rect_h = cv2.boundingRect(contour)
        
        aspect_ratio = rect_w / rect_h if rect_h > 0 else 0
        area_ratio = (rect_w * rect_h) / (w * h)
        
        if (2.0 <= aspect_ratio <= 5.0 and 
            0.02 <= area_ratio <= 0.3 and 
            rect_w >= 30 and rect_h >= 10):
            
            pad_x = max(5, int(rect_w * 0.1))
            pad_y = max(3, int(rect_h * 0.1))
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(w, x + rect_w + pad_x)
            y2 = min(h, y + rect_h + pad_y)
            
            plate_region = image_roi[y1:y2, x1:x2]
            if plate_region.size > 0:
                plate_candidates.append(plate_region)
    
    if not plate_candidates:
        bottom_portion = image_roi[int(h*0.7):h, :]
        if bottom_portion.size > 0:
            plate_candidates.append(bottom_portion)
        else:
            plate_candidates.append(image_roi)
    
    return plate_candidates

def is_valid_plate_text(text):
    """Enhanced validation for license plate text."""
    if not text or len(text) < 2:
        return False
    
    cleaned = text.replace(' ', '').replace('-', '').replace('_', '').upper()
    cleaned = ''.join(c for c in cleaned if c.isalnum())
    
    if len(cleaned) < 3 or len(cleaned) > 10:
        return False
    
    has_letter = any(c.isalpha() for c in cleaned)
    has_number = any(c.isdigit() for c in cleaned)
    
    letter_count = sum(1 for c in cleaned if c.isalpha())
    number_count = sum(1 for c in cleaned if c.isdigit())
    
    if has_letter and has_number:
        return True
    elif letter_count >= 3 and number_count <= 2:
        return True
    elif number_count >= 3 and letter_count <= 2:
        return True
    
    return False

def recognize_plate(image_roi):
    """Enhanced license plate recognition with better preprocessing and multiple attempts."""
    if image_roi is None or image_roi.size == 0:
        return "N/A"
    
    try:
        plate_regions = find_plate_regions(image_roi)
        
        best_result = ""
        best_confidence = 0.0
        
        for idx, region in enumerate(plate_regions):
            ocr_configs = [
                {'width_ths': 0.4, 'height_ths': 0.4, 'paragraph': True, 'decoder': 'greedy'},
                {'width_ths': 0.3, 'height_ths': 0.3, 'paragraph': False, 'decoder': 'greedy'},
                {'width_ths': 0.1, 'height_ths': 0.1, 'paragraph': True, 'decoder': 'beamsearch'}
            ]
            
            for config in ocr_configs:
                try:
                    results = plate_reader.readtext(
                        region, 
                        allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -',
                        **config
                    )
                    
                    full_text = ""
                    total_confidence = 0.0
                    valid_detections = 0
                    
                    results.sort(key=lambda x: x[0][0][0] if len(x[0]) > 0 else 0)
                    
                    for (bbox, text, confidence) in results:
                        cleaned_text = text.strip().upper()
                        if len(cleaned_text) > 0 and confidence > 0.2:
                            full_text += cleaned_text
                            total_confidence += confidence
                            valid_detections += 1
                    
                    if valid_detections > 0:
                        avg_confidence = total_confidence / valid_detections
                        full_text_cleaned = full_text.replace(' ', '').replace('-', '')
                        
                        if avg_confidence > best_confidence and is_valid_plate_text(full_text_cleaned):
                            best_result = full_text_cleaned
                            best_confidence = avg_confidence
                            
                except Exception:
                    continue
            
            try:
                preprocessed = preprocess_for_ocr(region)
                if preprocessed is not None:
                    for config in ocr_configs:
                        try:
                            results = plate_reader.readtext(
                                preprocessed,
                                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -',
                                **config
                            )
                            
                            full_text = ""
                            total_confidence = 0.0
                            valid_detections = 0
                            
                            results.sort(key=lambda x: x[0][0][0] if len(x[0]) > 0 else 0)
                            
                            for (bbox, text, confidence) in results:
                                cleaned_text = text.strip().upper()
                                if len(cleaned_text) > 0 and confidence > 0.2:
                                    full_text += cleaned_text
                                    total_confidence += confidence
                                    valid_detections += 1
                            
                            if valid_detections > 0:
                                avg_confidence = total_confidence / valid_detections
                                full_text_cleaned = full_text.replace(' ', '').replace('-', '')
                                
                                if avg_confidence > best_confidence and is_valid_plate_text(full_text_cleaned):
                                    best_result = full_text_cleaned
                                    best_confidence = avg_confidence
                                    
                        except Exception:
                            continue
                            
            except Exception:
                pass
        
        if best_confidence > 0.2 and best_result:
            return best_result
        
        return "N/A"
        
    except Exception as e:
        print(f"[OCR] Recognition error: {e}")
        return "N/A"

# ==============================
# Parking Spot Tracker
# ==============================
class ParkingSpotTracker:
    def __init__(self, spot_id, confirmation_time_occupied=10.0, confirmation_time_vacant=10.0, screenshot_delay=5.0):
        self.spot_id = spot_id
        self.is_occupied = False
        self.pending_status = None
        self.status_change_start = None
        self.confirmation_time_occupied = confirmation_time_occupied
        self.confirmation_time_vacant = confirmation_time_vacant
        self.last_detection_time = None
        self.confirmed_vehicle_info = None
        self.current_vehicle_roi = None
        
        # NEW: Screenshot delay tracking
        self.screenshot_delay = screenshot_delay
        self.screenshot_taken = False
        self.screenshot_pending_start = None
        self.pending_screenshot_info = None

    def update(self, detected_occupied, current_time, vehicle_info=None):
        if detected_occupied and vehicle_info:
            self.current_vehicle_roi = vehicle_info.get('image_roi')
        elif not detected_occupied:
            self.current_vehicle_roi = None
            
        if detected_occupied == self.is_occupied:
            self.pending_status = None
            self.status_change_start = None
            self.last_detection_time = current_time
            return self.is_occupied, None, None  # CHANGED: Added third return value
        
        if self.pending_status != detected_occupied:
            self.pending_status = detected_occupied
            self.status_change_start = current_time
            self.last_detection_time = current_time
            if detected_occupied:
                self.confirmed_vehicle_info = vehicle_info
            return self.is_occupied, None, None  # CHANGED: Added third return value
        
        self.last_detection_time = current_time
        confirmation_time = self.confirmation_time_occupied if detected_occupied else self.confirmation_time_vacant
        time_elapsed = current_time - self.status_change_start
        
        if time_elapsed >= confirmation_time:
            was_occupied = self.is_occupied
            self.is_occupied = detected_occupied
            self.pending_status = None
            self.status_change_start = None

            if self.is_occupied and not was_occupied:
                # CHANGED: Car just confirmed as parked - start screenshot delay timer
                self.screenshot_taken = False
                self.screenshot_pending_start = current_time
                self.pending_screenshot_info = self.confirmed_vehicle_info
                return self.is_occupied, self.confirmed_vehicle_info, None
            elif not self.is_occupied and was_occupied:
                # CHANGED: Car left - reset screenshot tracking
                self.screenshot_taken = False
                self.screenshot_pending_start = None
                self.pending_screenshot_info = None
                return self.is_occupied, self.spot_id, None

        return self.is_occupied, None, None  # CHANGED: Added third return value

    def check_screenshot_ready(self, current_time):
        """NEW METHOD: Check if enough time has passed to take the screenshot"""
        if (self.is_occupied and 
            not self.screenshot_taken and 
            self.screenshot_pending_start is not None and
            self.pending_screenshot_info is not None):
            
            time_since_confirmation = current_time - self.screenshot_pending_start
            
            if time_since_confirmation >= self.screenshot_delay:
                self.screenshot_taken = True
                info = self.pending_screenshot_info
                self.pending_screenshot_info = None
                return True, info
        
        return False, None

    def get_status_info(self, current_time):
        info = {
            'is_occupied': self.is_occupied,
            'pending_status': self.pending_status,
            'time_to_change': None,
            'is_pending': False,
            'screenshot_pending': not self.screenshot_taken and self.screenshot_pending_start is not None  # ADDED
        }
        if self.pending_status is not None and self.status_change_start is not None:
            confirmation_time = self.confirmation_time_occupied if self.pending_status else self.confirmation_time_vacant
            time_elapsed = current_time - self.status_change_start
            time_remaining = max(0, confirmation_time - time_elapsed)
            info['time_to_change'] = time_remaining
            info['is_pending'] = True
        return info


# ==============================
# Main Processing Loop for Single Video
# ==============================
def process_video_loop(config):
    """Process a single video feed."""
    try:
        cap = cv2.VideoCapture(config.video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video for lot {config.lot_id}: {config.video_path}", file=sys.stderr)
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        mask = load_parking_mask(config.mask_path, width, height)
        parking_spots = get_parking_spots(mask)
        total_spots = len(parking_spots)
        
        print(f"[LOT {config.lot_id}] Found {total_spots} parking spots")

        spot_trackers = {
            i: ParkingSpotTracker(
                i + 1, 
                OCCUPANCY_CONFIRMATION_TIME, 
                VACANCY_CONFIRMATION_TIME,
                SCREENSHOT_DELAY_AFTER_CONFIRMATION  # ADDED
            )
            for i, _ in enumerate(parking_spots)
        }

        start_time = time.time()
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_count += 1
            if frame_count % FRAME_SKIP_RATE != 0:
                continue

            current_time = time.time() - start_time
            loop_start = time.time()
            
            # YOLO detection
            processed_frame = cv2.resize(frame, (PROCESSING_WIDTH, PROCESSING_HEIGHT))
            results = model(processed_frame)
            
            vehicle_info_list = []
            try:
                det = results.pandas().xyxy[0]
                scale_x = width / PROCESSING_WIDTH
                scale_y = height / PROCESSING_HEIGHT

                for _, d in det.iterrows():
                    if int(d['class']) in VEHICLE_CLASSES:
                        x1 = int(float(d['xmin']) * scale_x)
                        y1 = int(float(d['ymin']) * scale_y)
                        x2 = int(float(d['xmax']) * scale_x)
                        y2 = int(float(d['ymax']) * scale_y)
                        
                        vehicle_roi = frame[y1:y2, x1:x2]
                        
                        vehicle_info_list.append({
                            'bbox': (x1, y1, x2, y2),
                            'type': VEHICLE_NAMES.get(int(d['class']), 'vehicle'),
                            'confidence': float(d['confidence']),
                            'image_roi': vehicle_roi
                        })
            except Exception as e:
                print(f"[LOT {config.lot_id}] YOLO detection error: {e}")

            # Draw annotations
            annotated_frame = frame.copy()
            
            # Add lot name label at top
            label_text = config.lot_name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            font_thickness = 3
            text_size = cv2.getTextSize(label_text, font, font_scale, font_thickness)[0]
            text_x = (width - text_size[0]) // 2
            text_y = 40
            
            # Draw background rectangle for better visibility
            cv2.rectangle(annotated_frame, (text_x - 10, text_y - text_size[1] - 10), 
                         (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
            cv2.putText(annotated_frame, label_text, (text_x, text_y), font, 
                       font_scale, (255, 255, 255), font_thickness)
            
            for veh in vehicle_info_list:
                x1, y1, x2, y2 = veh['bbox']
                label = f"{veh['type']}: {veh['confidence']:.2f}"
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Update spots and draw on frame
            occupied_count, pending_changes = 0, 0
            for i, spot in enumerate(parking_spots):
                occupying_vehicle = get_occupying_vehicle(spot, vehicle_info_list)
                detected_occupied = occupying_vehicle is not None
                tracker = spot_trackers[i]
                
                # CHANGED: Handle three return values
                confirmed_occupied, info, screenshot_info = tracker.update(detected_occupied, current_time, occupying_vehicle)

                # Handle parking/leaving events (no screenshot yet)
                if info is not None:
                    if confirmed_occupied:
                        log_car_parked(config, tracker.spot_id, info['type'], info['image_roi'])
                    else:
                        log_car_left(config, info)

                # NEW: Check if screenshot delay has elapsed
                screenshot_ready, screenshot_vehicle_info = tracker.check_screenshot_ready(current_time)
                if screenshot_ready and screenshot_vehicle_info is not None:
                    # Now take the screenshot and queue processing
                    queue_screenshot_processing(
                        config, 
                        tracker.spot_id, 
                        screenshot_vehicle_info['type'],
                        screenshot_vehicle_info['image_roi']
                    )

                if confirmed_occupied:
                    occupied_count += 1
                if tracker.get_status_info(current_time)['is_pending']:
                    pending_changes += 1

                x, y, w, h = spot
                color = (0, 0, 255) if confirmed_occupied else (0, 255, 0)
                cv2.rectangle(annotated_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(annotated_frame, f"P{i+1}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Update config data
            with config.data_lock:
                config.data['occupied'] = occupied_count
                config.data['total'] = total_spots
                config.data['pending'] = pending_changes
                config.data['last_update'] = datetime.now().strftime("%I:%M:%S %p")
                config.data['queue_size'] = processing_queue.qsize()
            
            loop_time = time.time() - loop_start
            print(f"[LOT {config.lot_id}] Frame processed in {loop_time:.3f}s | Occupied={occupied_count}/{total_spots} | Queue={processing_queue.qsize()}")

            # Update frame for this video
            with config.frame_lock:
                config.frame = annotated_frame.copy()
            
            # Update combined frame
            update_combined_frame()

            time.sleep(1/fps if fps > 0 else 0.03)

        cap.release()
    except Exception as e:
        print(f"[LOT {config.lot_id}] Error in video processing thread: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

# ==============================
# Combined Frame Generator
# ==============================
def update_combined_frame():
    """Combines all video frames into a single output."""
    global combined_frame
    
    frames_to_combine = []
    for config in video_configs:
        with config.frame_lock:
            if config.frame is not None:
                frames_to_combine.append(config.frame.copy())
    
    if not frames_to_combine:
        return
    
    # Arrange frames side by side or in a grid
    if len(frames_to_combine) == 1:
        combined = frames_to_combine[0]
    elif len(frames_to_combine) == 2:
        # Side by side
        h1, w1 = frames_to_combine[0].shape[:2]
        h2, w2 = frames_to_combine[1].shape[:2]
        max_h = max(h1, h2)
        
        # Resize to same height
        frame1 = cv2.resize(frames_to_combine[0], (int(w1 * max_h / h1), max_h))
        frame2 = cv2.resize(frames_to_combine[1], (int(w2 * max_h / h2), max_h))
        
        combined = np.hstack([frame1, frame2])
    else:
        # Grid layout for 3+ videos
        rows = []
        for i in range(0, len(frames_to_combine), 2):
            if i + 1 < len(frames_to_combine):
                h1, w1 = frames_to_combine[i].shape[:2]
                h2, w2 = frames_to_combine[i+1].shape[:2]
                max_h = max(h1, h2)
                
                frame1 = cv2.resize(frames_to_combine[i], (int(w1 * max_h / h1), max_h))
                frame2 = cv2.resize(frames_to_combine[i+1], (int(w2 * max_h / h2), max_h))
                
                rows.append(np.hstack([frame1, frame2]))
            else:
                rows.append(frames_to_combine[i])
        
        # Stack rows vertically
        combined = np.vstack(rows)
    
    with combined_frame_lock:
        combined_frame = combined

# ==============================
# Flask API Endpoints
# ==============================
@app.route('/status_display.html')
def status_display():
    """Serves the parking status display HTML page."""
    return send_from_directory('.', 'status_display.html')

@app.route('/data')
def get_data():
    """Returns the latest parking status data for all lots as a JSON array."""
    all_data = []
    for config in video_configs:
        with config.data_lock:
            all_data.append(config.data.copy())
    return jsonify(all_data)

@app.route('/parking_history')
def get_history():
    """Returns the combined parking history log from all lots."""
    all_history = []
    for config in video_configs:
        with config.log_lock:
            all_history.extend(list(config.history))
    
    # Sort by timestamp (most recent first)
    all_history.sort(key=lambda x: x.get('timestamp_in', ''), reverse=True)
    return jsonify(all_history[:50])  # Return last 50 entries

@app.route('/plate_screenshots/<filename>')
def serve_plate_screenshot(filename):
    """Serves plate screenshot images."""
    return send_from_directory(PLATE_SCREENSHOTS_DIR, filename)

def generate_frames():
    """Generates JPEG frames for the combined video stream."""
    while True:
        with combined_frame_lock:
            if combined_frame is None:
                time.sleep(0.1)
                continue
            try:
                ret, buffer = cv2.imencode('.jpg', combined_frame)
                if not ret:
                    time.sleep(0.1)
                    continue
                frame_bytes = buffer.tobytes()
            except Exception as e:
                print(f"Frame encoding error: {e}")
                time.sleep(0.1)
                continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    """Serves the combined real-time video stream as an MJPEG stream."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

# ==============================
# Entry Point
# ==============================
def main():
    print("=== ParkAlisto: Multi-Video Dashboard Server ===")
    print("🚀 NOW WITH MULTI-VIDEO SUPPORT!")
    
    root = Tk()
    root.withdraw()
    
    # Ask how many parking lots to monitor
    from tkinter import simpledialog
    num_lots = simpledialog.askinteger("Number of Parking Lots", 
                                       "How many parking lots do you want to monitor?",
                                       minvalue=1, maxvalue=10)
    if not num_lots:
        print("Number of lots is required. Exiting.", file=sys.stderr)
        return
    
    # Collect video and mask for each lot
    for i in range(num_lots):
        print(f"\n=== Setting up Parking Lot {i+1} ===")
        
        from tkinter import simpledialog
        lot_name = simpledialog.askstring("Lot Name", 
                                         f"Enter name for Parking Lot {i+1}:",
                                         initialvalue=f"Parking Lot {i+1}")
        if not lot_name:
            lot_name = f"Parking Lot {i+1}"
        
        video_path = askopenfilename(
            title=f"Select Video File for {lot_name}", 
            filetypes=[("Video Files", "*.mp4 *.avi *.mov"), ("All Files", "*.*")]
        )
        if not video_path:
            print(f"Video selection is required for {lot_name}. Exiting.", file=sys.stderr)
            return

        mask_path = askopenfilename(
            title=f"Select Mask Image for {lot_name}", 
            filetypes=[("Image Files", "*.png *.jpg *.jpeg"), ("All Files", "*.*")]
        )
        if not mask_path:
            print(f"Mask selection is required for {lot_name}. Exiting.", file=sys.stderr)
            return
        
        config = VideoConfig(i, lot_name, video_path, mask_path)
        video_configs.append(config)
        
        print(f"✓ {lot_name}")
        print(f"  Video: {video_path}")
        print(f"  Mask: {mask_path}")
    
    # Start background processing worker threads
    print(f"\n🔧 Starting {NUM_PROCESSING_THREADS} background processing threads...")
    worker_threads = []
    for i in range(NUM_PROCESSING_THREADS):
        worker = threading.Thread(target=background_processor_worker, name=f"Worker-{i+1}")
        worker.daemon = True
        worker.start()
        worker_threads.append(worker)
        print(f"   ✓ Worker thread {i+1} started")

    # Start video processing thread for each lot
    print(f"\n🎥 Starting video processing threads...")
    processing_threads = []
    for config in video_configs:
        thread = threading.Thread(
            target=process_video_loop, 
            args=(config,),
            name=f"VideoProcessor-Lot{config.lot_id}"
        )
        thread.daemon = True
        thread.start()
        processing_threads.append(thread)
        print(f"   ✓ Processing thread for {config.lot_name} started")

    print("\nWaiting for video processing to start...")
    while combined_frame is None:
        print(".", end="", flush=True)
        time.sleep(1)
    print(" Ready!")

    print("\n" + "="*60)
    print("🚀 MULTI-VIDEO SERVER IS RUNNING!")
    print("="*60)
    print(f"📍 Monitoring {len(video_configs)} parking lot(s):")
    for config in video_configs:
        print(f"   • {config.lot_name}")
    print("\n🌐 Access Points:")
    print("   📍 Dashboard: http://localhost:5000")
    print("   🎥 Combined Video: http://localhost:5000/video_feed")
    print("   📊 Data API: http://localhost:5000/data")
    print("   📝 History API: http://localhost:5000/parking_history")
    print("   🖼️  Screenshots: http://localhost:5000/plate_screenshots/<filename>")
    print("="*60)
    print("\n✨ HOW IT WORKS:")
    print("   1. All videos processed simultaneously in parallel")
    print("   2. Combined into single unified video stream")
    print("   3. Separate tracking for each parking lot")
    print("   4. Unified history log across all lots")
    print("   5. Background processing for plate/color detection")
    print("\n⚡ PERFORMANCE:")
    print(f"   • {len(video_configs)} video processing threads")
    print(f"   • {NUM_PROCESSING_THREADS} background worker threads")
    print("   • Real-time multi-lot monitoring")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == "__main__":
    main()