import unittest
from app import create_app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_status_endpoint(self):
        response = self.client.get('/system_status')
        self.assertEqual(response.status_code, 200)
        self.assertIn('configured', response.json)

if __name__ == '__main__':
    unittest.main()
