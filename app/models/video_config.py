from collections import deque
import threading

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
        # History limit: 100 entries
        self.history = deque(maxlen=100)
        self.active_log = {}
        self.frame_lock = threading.Lock()
        self.data_lock = threading.Lock()
        self.log_lock = threading.Lock()
