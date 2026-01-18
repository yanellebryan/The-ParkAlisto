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
                # download_enabled=False ensures it won't try to connect to the internet
                self.reader = easyocr.Reader(['en'], gpu=(self.device == 'cuda'), download_enabled=False)
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
            detected_parts = []
            for _, text, conf in results:
                # Keep text with decent confidence
                # Even short text (1-3 chars) might be part of the plate (e.g. "A 1")
                if conf > 0.3:
                    clean_text = text.upper().replace(' ', '')
                    if clean_text:
                        detected_parts.append(clean_text)
            
            if detected_parts:
                # Join all parts to form the full plate
                full_plate = "".join(detected_parts)
                # Ensure it has enough characters to be a valid plate (e.g. > 3 total)
                if len(full_plate) > 3:
                    return full_plate
                    
        except Exception as e:
            print(f"OCR error: {e}")
        return "N/A"

plate_recognizer = OCRService()
