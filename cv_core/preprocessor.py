import os
import cv2
import numpy as np

class ChequePreprocessor:
    def __init__(self):
        pass

    def load_image(self, image_path):
        """Loads an image from the given path. PDFs are rendered to an
        image (first page only) before the rest of the pipeline sees them,
        since OpenCV cannot decode PDF files directly."""
        if image_path.lower().endswith(".pdf"):
            return self._load_pdf_as_image(image_path)

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        return image

    def _load_pdf_as_image(self, pdf_path):
        """Renders the first page of a PDF to a BGR image array using PyMuPDF."""
        try:
            import pymupdf as fitz  # PyMuPDF (module renamed from `fitz` to `pymupdf`)
        except ImportError as e:
            raise RuntimeError(
                "PDF support requires PyMuPDF. Install it with: pip install PyMuPDF"
            ) from e

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Could not load PDF at {pdf_path}")

        doc = fitz.open(pdf_path)
        try:
            if doc.page_count == 0:
                raise FileNotFoundError(f"PDF at {pdf_path} has no pages")
            # Render at a higher DPI than the default 72 so text stays legible for OCR.
            pix = doc[0].get_pixmap(dpi=300)
            img_bytes = pix.tobytes("png")
        finally:
            doc.close()

        image = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not render PDF at {pdf_path}")
        return image

    def grayscale(self, image):
        """Converts an image to grayscale."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise(self, image):
        """Applies Gaussian blur to remove high-frequency noise."""
        return cv2.GaussianBlur(image, (5, 5), 0)

    def binarize(self, image):
        """
        Converts grayscale image to binary using Otsu's thresholding.
        Returns a binary image where text is typically black (0) and background is white (255).
        """
        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def deskew(self, image):
        """
        Detects the skew angle of the cheque and rotates it to be horizontal.
        Assumes the input is a binary image.
        """
        # Invert the image (so text is white, background is black) to find contours easily
        inverted = cv2.bitwise_not(image)
        
        # Find coordinates of all non-zero pixels
        coords = np.column_stack(np.where(inverted > 0))
        
        if len(coords) == 0:
            return image
            
        # Get the minimum bounding rectangle for the text
        angle = cv2.minAreaRect(coords)[-1]

        # `cv2.minAreaRect` returns values in the range [-90, 0)
        # We need to adjust the angle based on its orientation
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle

        # If the angle is very small, no rotation is needed
        if abs(angle) < 0.5:
            return image

        # Rotate the image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Perform the affine transformation, filling empty space with white
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated

    def preprocess(self, image_path):
        """
        Executes the full preprocessing pipeline.
        Returns the deskewed binarized image and the original image for reference.
        """
        original_img = self.load_image(image_path)
        gray_img = self.grayscale(original_img)
        denoised_img = self.remove_noise(gray_img)
        binary_img = self.binarize(denoised_img)
        deskewed_img = self.deskew(binary_img)
        
        return deskewed_img, original_img

# Example usage (commented out)
# if __name__ == "__main__":
#     preprocessor = ChequePreprocessor()
#     clean_img, orig_img = preprocessor.preprocess("sample_cheque.jpg")
#     cv2.imwrite("preprocessed_cheque.jpg", clean_img)
