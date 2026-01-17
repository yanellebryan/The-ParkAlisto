import threading
from queue import Queue

class SystemState:
    def __init__(self):
        self.is_configured = False
        self.is_running = False
        
        # Combined frame for dashboard
        self.combined_frame = None
        self.combined_frame_lock = threading.Lock()
        
        # Shared processing queue
        self.processing_queue = Queue()
        
        # Global list of video configurations
        self.video_configs = []

# Global instance
system_state = SystemState()
