import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.routes.api_routes import system_state
from config import Config
from app.models.video_config import VideoConfig
import unittest

class TestMonitoringSearch(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        
        # Setup mock data in system_state
        system_state.video_configs = []
        config = VideoConfig(0, "Test Lot", "test.mp4", "mask.png")
        # Mock data
        config.data['total'] = 10
        config.data['occupied'] = 5
        
        # Mock history
        history_entry = {
            'unique_id': 'test-123',
            'plate_number': 'ABC-123',
            'timestamp_in': '12:00:00 PM',
            'timestamp_out': 'Active',
            'plate_image': 'test_plate.jpg',
            'lot_name': 'Test Lot',
            'spot_id': 1,
            'vehicle_type': 'car',
            'color': 'red'
        }
        config.history.append(history_entry)
        
        system_state.video_configs.append(config)

    def test_global_stats(self):
        response = self.client.get('/stats/global')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        
        self.assertEqual(data['total_spots'], 10)
        self.assertEqual(data['total_occupied'], 5)
        self.assertEqual(data['total_available'], 5)
        self.assertEqual(data['total_parked_today'], 1)
        self.assertEqual(len(data['lot_stats']), 1)
        self.assertEqual(data['lot_stats'][0]['lot_name'], "Test Lot")

    def test_search_plate_found(self):
        response = self.client.get('/search?plate=ABC')
        self.assertEqual(response.status_code, 200)
        results = response.get_json()
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['plate_number'], 'ABC-123')

    def test_search_plate_not_found(self):
        response = self.client.get('/search?plate=XYZ')
        self.assertEqual(response.status_code, 200)
        results = response.get_json()
        
        self.assertEqual(len(results), 0)
        
    def test_monitoring_page_load(self):
        response = self.client.get('/monitoring')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Monitoring Center', response.data)

if __name__ == '__main__':
    unittest.main()
