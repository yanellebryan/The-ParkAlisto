import unittest
from unittest.mock import patch, MagicMock
from app import create_app
from config import Config
import io

class TestAddSource(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()
        self.app_context = self.app.app_context()
        self.app_context.push()

    def tearDown(self):
        self.app_context.pop()

    @patch('app.routes.api_routes.process_video_loop')
    @patch('app.routes.api_routes.background_processor_worker')
    @patch('app.routes.api_routes.system_state')
    def test_add_source_success_url(self, mock_state, mock_bg_worker, mock_loop):
        # Setup mock state
        mock_state.video_configs = []
        mock_state.is_running = False
        mock_state.combined_frame_lock = MagicMock()
        mock_state.combined_frame_lock.__enter__ = MagicMock()
        mock_state.combined_frame_lock.__exit__ = MagicMock()

        # Mock form data
        data = {
            'lot_name': 'Test Lot',
            'video_type': 'url',
            'video_url': 'rtsp://127.0.0.1:8554/test',
            'mask_file': (io.BytesIO(b"fake image data"), 'mask.jpg')
        }

        response = self.client.post('/add_source', data=data, content_type='multipart/form-data')
        
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json['success'])
        self.assertEqual(response.json['lot_id'], 0)
        
        # Verify config was added
        self.assertEqual(len(mock_state.video_configs), 1)
        self.assertEqual(mock_state.video_configs[0].lot_name, 'Test Lot')
        
        # Verify threads started
        mock_loop.assert_called_once()
        # Should start bg workers if not running
        self.assertTrue(mock_state.is_running)

    @patch('app.routes.api_routes.process_video_loop')
    @patch('app.routes.api_routes.system_state')
    def test_add_source_missing_fields(self, mock_state, mock_loop):
        # Mock state
        mock_state.video_configs = []
        
        # Missing lot name
        data = {
            'video_type': 'url',
            'video_url': 'rtsp://test'
        }
        response = self.client.post('/add_source', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        
        # Missing mask
        data = {
            'lot_name': 'Test',
            'video_type': 'url',
            'video_url': 'rtsp://test'
        }
        response = self.client.post('/add_source', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
