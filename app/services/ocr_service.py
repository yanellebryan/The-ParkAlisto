import easyocr
import torch

class OCRService:
    def __init__(self):
        self.reader = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load_model(self):
        if self.reader is None:
            try:
                print(f"Loading EasyOCR (Device: {self.device})...")
                self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'))
                print("✓ EasyOCR loaded successfully")
            except Exception as e:
                print(f"Error loading EasyOCR: {e}")

    def recognize_plate(self, image_roi):
        if self.reader is None: 
            self.load_model()
            if self.reader is None:
                return "OCR_ERR"
        
        if image_roi is None or image_roi.size == 0: 
            return "N/A"
            
        try:
            results = self.reader.readtext(image_roi)
            for _, text, conf in results:
                if conf > 0.3 and len(text) > 3:
                    return text.upper().replace(' ', '')
        except Exception as e:
            print(f"OCR error: {e}")
        return "N/A"

plate_recognizer = OCRService()
