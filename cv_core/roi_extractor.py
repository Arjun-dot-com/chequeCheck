import os
import cv2
from ultralytics import YOLO

class ROIExtractor:
    def __init__(self, model_path=None):
        """
        Initializes the ROI Extractor using a trained YOLO model.
        """
        if model_path is None:
            # Default path after running train_yolo.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, "models", "cheque_roi_extractor", "weights", "best.pt")
            
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO model not found at {model_path}. Please train the model first.")
            
        self.model = YOLO(model_path)
        
        # YOLO classes from dataset.yaml
        self.class_names = [
            "IssueBank",    # 0
            "ReceiverName", # 1 (Payee)
            "AcNo",         # 2 (Account Number)
            "Amt",          # 3 (Amount)
            "ChqNo",        # 4 (Cheque Number)
            "DateIss",      # 5 (Date Issued)
            "Sign"          # 6 (Signature)
        ]

    def extract_all(self, image):
        """
        Runs the YOLO model on the image and extracts all recognized regions.
        Returns a dictionary of {class_name: cropped_image_array}.
        """
        # Run inference
        results = self.model(image, verbose=False)[0]
        
        rois = {}
        
        # Iterate over detected boxes
        for box in results.boxes:
            # Get class ID and confidence
            cls_id = int(box.cls[0].item())
            conf = box.conf[0].item()
            
            # We only extract if confidence is decent (e.g. > 0.3)
            if conf > 0.3:
                # Get integer bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Skip degenerate boxes (can happen due to coordinate rounding)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop the image
                cropped = image[y1:y2, x1:x2]

                # Store it using the class name
                class_name = self.class_names[cls_id]
                rois[class_name] = cropped
                
        return rois
