import cv2
import numpy as np

class FraudChecker:
    def __init__(self):
        pass

    def isolate_signature(self, signature_roi):
        """
        Takes the signature ROI and cleans it up.
        Returns a binary mask of the signature itself, ignoring background noise/lines.
        """
        if signature_roi is None or signature_roi.size == 0:
            return None, False

        # Convert to grayscale if not already
        if len(signature_roi.shape) == 3:
            gray = cv2.cvtColor(signature_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = signature_roi.copy()
            
        # Apply adaptive thresholding to handle variations in pen ink and lighting
        # We invert it so the signature (ink) is white (255) and background is black (0)
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Remove small noise (like printed dots or small background lines)
        kernel = np.ones((2,2), np.uint8)
        clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Optional: Check if the signature area is completely empty
        ink_pixels = cv2.countNonZero(clean)
        total_pixels = clean.shape[0] * clean.shape[1]
        ink_ratio = ink_pixels / total_pixels
        
        # If ink ratio is extremely low, it might be an unsigned cheque
        is_signed = ink_ratio > 0.01 
        
        return clean, is_signed

    def detect_tampering(self, roi_image):
        """
        Performs basic tampering detection.
        Looks for unnatural edge densities or high-frequency noise that might 
        indicate manual alteration, copy-pasting, or washing.
        Returns a tampering score (higher means more likely tampered).
        """
        if roi_image is None or roi_image.size == 0:
            return {"laplacian_variance": 0.0, "edge_density": 0.0, "flagged": False}

        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image.copy()

        # Compute the Laplacian variance which is often used for blur detection,
        # but extreme variations in small local regions can indicate splicing.
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Use Canny edge detection to find the number of sharp edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = cv2.countNonZero(edges) / (gray.shape[0] * gray.shape[1])
        
        # This is a very basic heuristic. Real tampering detection requires 
        # analyzing the micro-printing background or using deep learning anomalies.
        # For now, we return these raw metrics.
        
        metrics = {
            "laplacian_variance": laplacian_var,
            "edge_density": edge_density,
            # If edge density is abnormally high in a supposed blank area, flag it
            "flagged": edge_density > 0.15 
        }
        
        return metrics
