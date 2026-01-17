import time
import os
import cv2
import threading
from app.core import system_state
from config import Config
from .ocr_service import plate_recognizer
from .color_detector import color_detector

def background_processor_worker():
    """Worker thread that processes plate and color detection in the background."""
    print(f"[WORKER] Background processor thread started (ID: {threading.get_ident()})")
    
    while True:
        try:
            task = system_state.processing_queue.get()
            
            if task is None:
                break
            
            # Save screenshot
            # Use current time for filename
            screenshot_filename = f"lot{task.lot_id}_spot{task.spot_id}_{int(time.time())}.jpg"
            
            screenshot_path = os.path.join(Config.PLATE_SCREENSHOTS_DIR, screenshot_filename)
            try:
                cv2.imwrite(screenshot_path, task.vehicle_roi)
                task.screenshot_filename = screenshot_filename
            except Exception as e:
                print(f"[WORKER] Error saving screenshot: {e}")
            
            # Detect plate number
            plate_number = plate_recognizer.recognize_plate(task.vehicle_roi)
            
            # Detect color
            color = color_detector.get_vehicle_color(task.vehicle_roi)
            
            # Update the parking log
            # We need to access the video_config corresponding to the lot_id
            # Assuming lot_id is index in video_configs or we find it
            # In parkalisto.py, lot_id is index.
            if 0 <= task.lot_id < len(system_state.video_configs):
                config = system_state.video_configs[task.lot_id]
                update_processing_results(config, task.spot_id, plate_number, color, task.screenshot_filename)
            
            system_state.processing_queue.task_done()
            
        except Exception as e:
            print(f"[WORKER] Error in background processor: {e}")
            # traceback.print_exc()
            try:
                system_state.processing_queue.task_done()
            except:
                pass

def update_processing_results(config, spot_id, plate_number, color, screenshot_filename):
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
