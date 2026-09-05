import pytesseract
from pytesseract import Output
import cv2

class OCREngine:
    def __init__(self):
        # Configuration for specific fields to improve accuracy
        self.configs = {
            "micr": "--psm 7 -c tessedit_char_whitelist=0123456789abcd",
            "amount_num": "--psm 7 -c tessedit_char_whitelist=0123456789.,*",
            "date": "--psm 7",
            "payee": "--psm 7",
            "chq_no": "--psm 7 -c tessedit_char_whitelist=0123456789",
            "bank_name": "--psm 7",
        }

    def _read_text_and_conf(self, image, config):
        """Reads text and returns it along with the average confidence score."""
        try:
            data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
            
            words = []
            confidences = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = data['conf'][i]
                
                # Filter out empty strings and invalid confidences (-1)
                if text and int(conf) != -1:
                    words.append(text)
                    confidences.append(int(conf))
            
            final_text = " ".join(words)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                "value": final_text,
                "confidence": round(avg_conf, 2)
            }
        except Exception as e:
            return {"value": "", "confidence": 0.0, "error": str(e)}

    def extract_all_text(self, rois):
        """Takes a dictionary of {field_name: cropped_image} and returns
        {field_name: {"value": str, "confidence": float}} for each field
        that has a matching OCR config and a non-empty crop."""
        extracted_data = {}

        for field_name, image in rois.items():
            if field_name not in self.configs:
                continue
            if image is None or image.size == 0:
                continue
            extracted_data[field_name] = self._read_text_and_conf(image, self.configs[field_name])

        return extracted_data
