import time
import cv2
import numpy as np
import os
from datetime import datetime
from config import Config
from app.core import system_state
from app.models import ParkingSpotTracker, ProcessingTask
from app.utils.parking_helpers import load_parking_mask, get_parking_spots, get_occupying_vehicle
from app.utils.ffmpeg_capture import FFmpegVideoCapture
from .ai_detector import vehicle_detector

def process_video_loop(config):
    """Process a single video feed."""
    print(f"[Started] Processing thread for {config.lot_name} (Source: {config.video_path})")
    
    try:
        video_source = config.video_path
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)
            
        print(f"DEBUG: Attempting to open video source: {video_source}")
        
        cap = None
        is_rtsp = isinstance(video_source, str) and video_source.startswith('rtsp')
        
        if is_rtsp:
            print("DEBUG: Detected RTSP stream, using FFmpegVideoCapture backend.")
            cap = FFmpegVideoCapture(video_source)
        else:
            print("DEBUG: Detected file/device, using cv2.VideoCapture.")
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video for lot {config.lot_id}: {video_source}")
            return
        else:
            print(f"DEBUG: Successfully opened video source: {video_source}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"[{config.lot_name}] Video: {width}x{height} @ {fps}fps")

        mask = load_parking_mask(config.mask_path, width, height)
        parking_spots = get_parking_spots(mask)
        total_spots = len(parking_spots)
        print(f"[{config.lot_name}] Detected {total_spots} parking spots")
        
        spot_trackers = {
            i: ParkingSpotTracker(
                i + 1,
                Config.OCCUPANCY_CONFIRMATION_TIME,
                Config.VACANCY_CONFIRMATION_TIME,
                Config.SCREENSHOT_DELAY_AFTER_CONFIRMATION
            )
            for i, _ in enumerate(parking_spots)
        }

    except Exception as e:
        print(f"CRITICAL ERROR in process_video_loop for {config.lot_name}: {e}")
        import traceback
        traceback.print_exc()
        return

    start_time = time.time()
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # If it's a file, loop. If it's RTSP, break (auto-reconnect logic could go here)
            if not is_rtsp:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                print(f"[{config.lot_name}] RTSP stream ended or failed.")
                break

        frame_count += 1
        if frame_count % Config.FRAME_SKIP_RATE != 0:
            continue

        current_time = time.time() - start_time
        
        # YOLO detection
        processed_frame = cv2.resize(frame, (Config.PROCESSING_WIDTH, Config.PROCESSING_HEIGHT))
        
        # AI Detection
        # Pass the processed frame
        # The detector expects frame. 
        # But wait, ai_detector.detect() returns boxes in processed_frame coordinates if we pass processed_frame?
        # Yes, standard YOLO behavior.
        detection_results = vehicle_detector.detect(processed_frame)
        
        vehicle_info_list = []
        scale_x = width / Config.PROCESSING_WIDTH
        scale_y = height / Config.PROCESSING_HEIGHT

        for d in detection_results:
            x1 = int(d['xmin'] * scale_x)
            y1 = int(d['ymin'] * scale_y)
            x2 = int(d['xmax'] * scale_x)
            y2 = int(d['ymax'] * scale_y)
            
            vehicle_info_list.append({
                'bbox': (x1, y1, x2, y2),
                'type': d['type'],
                'confidence': d['confidence'],
                'image_roi': frame[y1:y2, x1:x2]
            })

        # Draw annotations
        annotated_frame = frame.copy()
        
        occupied_count = 0
        pending_changes = 0
        
        for i, spot in enumerate(parking_spots):
            occupying_vehicle = get_occupying_vehicle(spot, vehicle_info_list)
            tracker = spot_trackers[i]
            
            confirmed_occupied, info, _ = tracker.update(
                occupying_vehicle is not None, 
                current_time, 
                occupying_vehicle
            )

            if info is not None:
                # Log status change
                if confirmed_occupied:
                    log_car_parked(config, tracker.spot_id, info['type'], info['image_roi'])
                else:
                    log_car_left(config, info) # info is spot_id when leaving

            screenshot_ready, screenshot_vehicle_info = tracker.check_screenshot_ready(current_time)
            if screenshot_ready:
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
            cv2.putText(
                annotated_frame, 
                f"P{i+1}", 
                (x, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                color, 
                1
            )

        # Update global data
        with config.data_lock:
            config.data.update({
                'occupied': occupied_count,
                'total': total_spots,
                'pending': pending_changes,
                'last_update': datetime.now().strftime("%I:%M:%S %p"),
                'queue_size': system_state.processing_queue.qsize()
            })

        with config.frame_lock:
            config.frame = annotated_frame.copy()
        
        update_combined_frame()
        time.sleep(0.01)

def update_combined_frame():
    # DISABLING stitching to prevent crashes and enable client-side grid view
    # frames = []
    return
    
    # Logic below is disabled
    # for config in system_state.video_configs:
    #     with config.frame_lock:
    #         if config.frame is not None:
    #             frames.append(config.frame.copy())
    # ...

# Logging helpers local to this service or imported?
# They manipulate config state which is passed.
# They enqueue tasks to system_state.processing_queue.

def log_car_parked(config, spot_id, vehicle_type, vehicle_roi=None):
    timestamp = datetime.now().strftime("%I:%M:%S %p %B %d, %Y")
    with config.log_lock:
        log_entry = {
            'unique_id': f"{config.lot_id}_{spot_id}_{int(time.time())}",
            'lot_id': config.lot_id,
            'lot_name': config.lot_name,
            'spot_id': spot_id,
            'plate_number': 'Waiting...',
            'color': 'Waiting...',
            'vehicle_type': vehicle_type,
            'timestamp_in': timestamp,
            'is_active': True,
            'plate_image': None,
            'processing_status': 'waiting_for_screenshot'
        }
        config.active_log[spot_id] = log_entry
        config.history.appendleft(log_entry.copy())
    
    if vehicle_roi is not None:
        task = ProcessingTask(config.lot_id, spot_id, vehicle_roi.copy(), vehicle_type, timestamp)
        system_state.processing_queue.put(task)

def queue_screenshot_processing(config, spot_id, vehicle_type, vehicle_roi):
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
    system_state.processing_queue.put(task)

def log_car_left(config, spot_id):
    with config.log_lock:
        if spot_id in config.active_log:
            log_entry = config.active_log[spot_id]
            log_entry['timestamp_out'] = datetime.now().strftime("%I:%M:%S %p %B %d, %Y")
            log_entry['is_active'] = False
            del config.active_log[spot_id]
