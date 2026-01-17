from config import Config

class ParkingSpotTracker:
    def __init__(self, spot_id, confirmation_time_occupied=None, confirmation_time_vacant=None, screenshot_delay=None):
        self.spot_id = spot_id
        self.is_occupied = False
        self.pending_status = None
        self.status_change_start = None
        self.confirmation_time_occupied = confirmation_time_occupied or Config.OCCUPANCY_CONFIRMATION_TIME
        self.confirmation_time_vacant = confirmation_time_vacant or Config.VACANCY_CONFIRMATION_TIME
        self.last_detection_time = None
        self.confirmed_vehicle_info = None
        
        self.screenshot_delay = screenshot_delay or Config.SCREENSHOT_DELAY_AFTER_CONFIRMATION
        self.screenshot_taken = False
        self.screenshot_pending_start = None
        self.pending_screenshot_info = None

    def update(self, detected_occupied, current_time, vehicle_info=None):
        if detected_occupied == self.is_occupied:
            self.pending_status = None
            self.status_change_start = None
            return self.is_occupied, None, None
        
        if self.pending_status != detected_occupied:
            self.pending_status = detected_occupied
            self.status_change_start = current_time
            if detected_occupied:
                self.confirmed_vehicle_info = vehicle_info
            return self.is_occupied, None, None
        
        confirmation_time = self.confirmation_time_occupied if detected_occupied else self.confirmation_time_vacant
        time_elapsed = current_time - self.status_change_start
        
        if time_elapsed >= confirmation_time:
            was_occupied = self.is_occupied
            self.is_occupied = detected_occupied
            self.pending_status = None
            self.status_change_start = None

            if self.is_occupied and not was_occupied:
                self.screenshot_taken = False
                self.screenshot_pending_start = current_time
                self.pending_screenshot_info = self.confirmed_vehicle_info
                return self.is_occupied, self.confirmed_vehicle_info, None
            elif not self.is_occupied and was_occupied:
                self.screenshot_taken = False
                self.screenshot_pending_start = None
                self.pending_screenshot_info = None
                return self.is_occupied, self.spot_id, None

        return self.is_occupied, None, None

    def check_screenshot_ready(self, current_time):
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
            'is_pending': False
        }
        if self.pending_status is not None and self.status_change_start is not None:
            confirmation_time = self.confirmation_time_occupied if self.pending_status else self.confirmation_time_vacant
            time_elapsed = current_time - self.status_change_start
            info['time_to_change'] = max(0, confirmation_time - time_elapsed)
            info['is_pending'] = True
        return info
