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

@web_bp.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@web_bp.route('/plate_screenshots/<filename>')
def serve_plate_screenshot(filename):
    return send_from_directory(Config.PLATE_SCREENSHOTS_DIR, filename)
