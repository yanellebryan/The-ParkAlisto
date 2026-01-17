import unittest
from app.services.ocr_service import plate_recognizer

class TestOCR(unittest.TestCase):
    def test_ocr_loading(self):
        # plate_recognizer.load_model()
        # self.assertIsNotNone(plate_recognizer.reader)
        pass

if __name__ == '__main__':
    unittest.main()
