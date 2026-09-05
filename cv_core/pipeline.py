import os
import cv2
from .preprocessor import ChequePreprocessor
from .roi_extractor import ROIExtractor
from .ocr_engine import OCREngine
from .fraud_checker import FraudChecker

class ChequeProcessingPipeline:
    def __init__(self):
        self.preprocessor = ChequePreprocessor()
        self.roi_extractor = ROIExtractor()
        self.ocr_engine = OCREngine()
        self.fraud_checker = FraudChecker()

    def process_cheque(self, image_path):
        """
        Runs the full CV and OCR pipeline on a cheque image.
        Returns a dictionary containing extracted data and fraud metrics.
        """
        if not os.path.exists(image_path):
            return {"error": f"Image not found at {image_path}"}

        # 1. Preprocessing
        deskewed_binary, original_img = self.preprocessor.preprocess(image_path)
        
        # 2. ROI Extraction
        # Extract ROIs using the YOLO model on the original image (color is fine for YOLO)
        rois = self.roi_extractor.extract_all(original_img)
        
        # Map YOLO class names to the field names the OCR engine has configs for.
        mapped_rois = {}
        if "DateIss" in rois: mapped_rois["date"] = rois["DateIss"]
        if "ReceiverName" in rois: mapped_rois["payee"] = rois["ReceiverName"]
        if "Amt" in rois: mapped_rois["amount_num"] = rois["Amt"]
        if "ChqNo" in rois: mapped_rois["chq_no"] = rois["ChqNo"]
        if "IssueBank" in rois: mapped_rois["bank_name"] = rois["IssueBank"]

        # The MICR band usually contains AcNo and ChqNo. If the model detects them separately,
        # we can pass them as 'micr' one by one, or update OCR engine. For now, let's just
        # pass the Account Number to be read using MICR config.
        if "AcNo" in rois: mapped_rois["micr"] = rois["AcNo"]
        
        # 3. OCR Extraction
        ocr_results = self.ocr_engine.extract_all_text(mapped_rois)
        
        # 4. Fraud Detection
        # Check signature presence
        is_signed = False
        if rois.get("Sign") is not None and rois["Sign"].size > 0:
            sig_clean, is_signed = self.fraud_checker.isolate_signature(rois["Sign"])

        # Check tampering on amount and payee fields
        tamper_amount = {}
        if rois.get("Amt") is not None and rois["Amt"].size > 0:
            tamper_amount = self.fraud_checker.detect_tampering(rois["Amt"])

        tamper_payee = {}
        if rois.get("ReceiverName") is not None and rois["ReceiverName"].size > 0:
            tamper_payee = self.fraud_checker.detect_tampering(rois["ReceiverName"])
        
        # Helper to safely format OCR results
        def format_ocr(key):
            res = ocr_results.get(key)
            if isinstance(res, dict):
                return res
            return {"value": "", "confidence": 0.0}

        # Compile final results
        result = {
            "status": "success",
            "extracted_data": {
                "micr_code": format_ocr("micr"),
                "date": format_ocr("date"),
                "payee": format_ocr("payee"),
                "amount": format_ocr("amount_num"),
                "cheque_number": format_ocr("chq_no"),
                "bank_name": format_ocr("bank_name")
            },
            "validation": {
                "is_signed": is_signed,
                "amount_tamper_flag": tamper_amount.get("flagged", False),
                "payee_tamper_flag": tamper_payee.get("flagged", False),
            }
        }
        
        return result

# Example usage:
# if __name__ == "__main__":
#     pipeline = ChequeProcessingPipeline()
#     results = pipeline.process_cheque("path/to/cheque.jpg")
#     print(results)
