from flask import render_template, Response, send_from_directory, current_app
from . import web_bp
from app.core import system_state
import cv2
import time
from config import Config

def generate_frames():
    while True:
        with system_state.combined_frame_lock:
            if system_state.combined_frame is None:
                time.sleep(0.1)
                continue
            try:
                ret, buffer = cv2.imencode('.jpg', system_state.combined_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret: 
                    continue
                frame_bytes = buffer.tobytes()
            except Exception as e:
                print(f"Frame encode error: {e}")
                continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.05)

@web_bp.route('/')
def index():
    return render_template('index.html')

@web_bp.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')

@web_bp.route('/entrance')
def entrance():
    return render_template('entrance.html')

@web_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@web_bp.route('/plate_screenshots/<filename>')
def serve_plate_screenshot(filename):
    return send_from_directory(Config.PLATE_SCREENSHOTS_DIR, filename)

def generate_frames_single(lot_id):
    """Generator for a specific camera stream."""
    while True:
        # Find the config for this lot_id
        target_config = None
        for config in system_state.video_configs:
            if config.lot_id == lot_id:
                target_config = config
                break
        
        if not target_config:
            time.sleep(1.0)
            continue
            
        with target_config.frame_lock:
            if target_config.frame is None:
                time.sleep(0.01)
                continue
            
            try:
                ret, buffer = cv2.imencode('.jpg', target_config.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()
            except Exception as e:
                print(f"Frame encode error for lot {lot_id}: {e}")
                continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # Cap at ~30fps for browser

@web_bp.route('/video_feed/<int:lot_id>')
def video_feed_single(lot_id):
    return Response(generate_frames_single(lot_id), mimetype='multipart/x-mixed-replace; boundary=frame')

