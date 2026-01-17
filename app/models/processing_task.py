class ProcessingTask:
    def __init__(self, lot_id, spot_id, vehicle_roi, vehicle_type, timestamp):
        self.lot_id = lot_id
        self.spot_id = spot_id
        self.vehicle_roi = vehicle_roi
        self.vehicle_type = vehicle_type
        self.timestamp = timestamp
        self.screenshot_filename = None
