"""End-to-end orchestration for automated cheque image processing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .fraud_checker import DEFAULT_SNN_MODEL_PATH, FraudChecker
from .ocr_engine import DEFAULT_TROCR_MODEL, OCREngine
from .preprocessor import ChequePreprocessor
from .roi_extractor import ROIExtractor


YOLO_TO_API_FIELD: dict[str, str] = {
    "IssueBank": "bank_name",
    "ReceiverName": "payee",
    "AcNo": "micr_code",
    "Amt": "amount",
    "ChqNo": "cheque_number",
    "DateIss": "date",
}

EXTRACTED_FIELDS: tuple[str, ...] = (
    "micr_code",
    "date",
    "payee",
    "amount",
    "cheque_number",
    "bank_name",
)


class ChequeProcessingPipeline:
    """Coordinate preprocessing, ROI detection, OCR, and SNN verification.

    All model-backed components are constructed once in ``__init__``. Reusing
    one pipeline instance therefore avoids loading YOLO, TrOCR, or the Siamese
    network during individual API requests.
    """

    def __init__(
        self,
        roi_model_path: str | Path | None = None,
        trocr_model_name: str = DEFAULT_TROCR_MODEL,
        snn_model_path: str | Path = DEFAULT_SNN_MODEL_PATH,
        snn_distance_threshold: float = 0.5,
    ) -> None:
        """Initialize each processing component exactly once.

        Args:
            roi_model_path: Optional override for the trained YOLO checkpoint.
            trocr_model_name: Hugging Face identifier or local TrOCR directory.
            snn_model_path: Path to the trained Siamese-network checkpoint.
            snn_distance_threshold: Maximum SNN distance considered a match.
        """
        self.preprocessor = ChequePreprocessor()
        self.roi_extractor = ROIExtractor(
            str(roi_model_path) if roi_model_path is not None else None
        )
        self.ocr_engine = OCREngine(model_name=trocr_model_name)
        self.fraud_checker = FraudChecker(
            model_path=snn_model_path,
            distance_threshold=snn_distance_threshold,
        )

    @staticmethod
    def _load_reference(reference_path: str | None) -> np.ndarray | None:
        """Load an optional reference image with OpenCV.

        Missing, corrupt, and undecodable references return ``None`` so the
        fraud methods can produce their documented unverified defaults.
        """
        if reference_path is None:
            return None
        path = Path(reference_path)
        if not path.is_file():
            return None
        return cv2.imread(str(path), cv2.IMREAD_COLOR)

    @staticmethod
    def _empty_ocr_result() -> dict[str, str | float]:
        """Return the frontend placeholder for an undetected field."""
        return {"value": "", "confidence": 0.0}

    def process_cheque(
        self,
        image_path: str,
        reference_signature_path: str | None = None,
        reference_amount_path: str | None = None,
    ) -> dict[str, object]:
        """Process one cheque and return the stable frontend response schema.

        Processing order is preprocessing, YOLO ROI extraction, TrOCR text
        recognition, and finally shared-SNN signature/amount comparison.

        Args:
            image_path: Path to the uploaded cheque image or single-page PDF.
            reference_signature_path: Optional genuine signature image path.
            reference_amount_path: Optional visual reference for the expected
                handwritten amount.

        Returns:
            Dictionary containing exactly ``status``, ``extracted_data``, and
            ``validation`` at the top level.

        Raises:
            FileNotFoundError: If the cheque path does not exist or cannot be
                decoded by the preprocessor.
        """
        if not Path(image_path).is_file():
            raise FileNotFoundError(f"Cheque image not found: {image_path}")

        # The cleaned image remains available for future preprocessing-aware
        # detectors; YOLO currently operates on the original color image.
        _cleaned_image, original_image = self.preprocessor.preprocess(image_path)
        rois = self.roi_extractor.extract_all(original_image)

        mapped_rois = {
            api_field: rois[yolo_field]
            for yolo_field, api_field in YOLO_TO_API_FIELD.items()
            if yolo_field in rois and rois[yolo_field].size > 0
        }
        recognized_fields = self.ocr_engine.extract_all_text(mapped_rois)
        extracted_data = {
            field: recognized_fields.get(field, self._empty_ocr_result())
            for field in EXTRACTED_FIELDS
        }

        is_signed = False
        amount_tamper_flag = False

        reference_signature = self._load_reference(reference_signature_path)
        signature_crop = rois.get("Sign")
        if reference_signature is not None and signature_crop is not None:
            signature_result = self.fraud_checker.check_signature_authenticity(
                signature_crop, reference_signature
            )
            is_signed = bool(signature_result["is_signed"])

        reference_amount = self._load_reference(reference_amount_path)
        amount_crop = rois.get("Amt")
        if reference_amount is not None and amount_crop is not None:
            amount_result = self.fraud_checker.spot_cursive_amount(
                amount_crop, reference_amount
            )
            amount_tamper_flag = bool(amount_result["amount_tamper_flag"])

        return {
            "status": "success",
            "extracted_data": extracted_data,
            "validation": {
                "is_signed": is_signed,
                "amount_tamper_flag": amount_tamper_flag,
                # No reference-payee input exists in the current API contract.
                "payee_tamper_flag": False,
            },
        }
