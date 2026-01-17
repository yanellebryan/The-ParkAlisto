import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev_default_key')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv('UPLOAD_FOLDER', 'uploads'))
    PLATE_SCREENSHOTS_DIR = os.path.join(BASE_DIR, os.getenv('PLATE_SCREENSHOTS_DIR', 'plate_screenshots'))
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

    # Processing
    NUM_PROCESSING_THREADS = int(os.getenv('NUM_PROCESSING_THREADS', 3))
    FRAME_SKIP_RATE = int(os.getenv('FRAME_SKIP_RATE', 10))
    PROCESSING_WIDTH = int(os.getenv('PROCESSING_WIDTH', 416))
    PROCESSING_HEIGHT = int(os.getenv('PROCESSING_HEIGHT', 416))
    
    # Confirmation Timers
    OCCUPANCY_CONFIRMATION_TIME = 10.0
    VACANCY_CONFIRMATION_TIME = 10.0
    SCREENSHOT_DELAY_AFTER_CONFIRMATION = 5.0
    
    # Vehicle Classes (COCO)
    VEHICLE_CLASSES = [2, 3, 5, 7]
    VEHICLE_NAMES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
