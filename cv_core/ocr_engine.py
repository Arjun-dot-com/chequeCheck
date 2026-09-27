"""Transformer-based optical character recognition for cheque fields."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_TROCR_MODEL = "microsoft/trocr-base-handwritten"


class OCREngine:
    """Recognize text in YOLO-extracted cheque fields with Microsoft TrOCR.

    An application should construct one ``OCREngine`` during process startup and
    reuse it for all requests. The processor and model are therefore loaded
    once by the engine constructor rather than once per crop.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TROCR_MODEL,
        device: str | torch.device | None = None,
    ) -> None:
        """Load the TrOCR processor and model for inference.

        Args:
            model_name: Hugging Face model identifier or local model directory.
            device: Optional explicit inference device. When omitted, CUDA is
                selected if available and CPU is used otherwise.
        """
        self.device = torch.device(device) if device is not None else DEVICE
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _to_pil_rgb(crop: np.ndarray) -> Image.Image:
        """Convert a grayscale, BGR, or BGRA OpenCV crop to an RGB PIL image.

        Args:
            crop: Non-empty image array supplied by the ROI extractor.

        Returns:
            An RGB ``PIL.Image.Image`` suitable for the TrOCR processor.

        Raises:
            ValueError: If the crop has an unsupported shape.
        """
        if crop.ndim == 2:
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.ndim == 3 and crop.shape[2] == 1:
            rgb = cv2.cvtColor(crop[:, :, 0], cv2.COLOR_GRAY2RGB)
        elif crop.ndim == 3 and crop.shape[2] == 3:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        elif crop.ndim == 3 and crop.shape[2] == 4:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        else:
            raise ValueError("ROI must be a grayscale, BGR, or BGRA image")

        return Image.fromarray(np.ascontiguousarray(rgb))

    def _recognize(self, crop: np.ndarray) -> str:
        """Generate text for one non-empty OpenCV image crop."""
        image = self._to_pil_rgb(crop)
        processor_output: Any = self.processor(images=image, return_tensors="pt")
        pixel_values = processor_output.pixel_values.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(pixel_values)

        return self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

    def extract_all_text(
        self, mapped_rois: dict[str, np.ndarray]
    ) -> dict[str, dict[str, str | float]]:
        """Recognize every mapped ROI and return the frontend field schema.

        Args:
            mapped_rois: Mapping from API field names to OpenCV image crops.

        Returns:
            Mapping of each input field to ``{"value": str, "confidence":
            100.0}``. TrOCR's standard greedy generation does not expose a
            calibrated word-level confidence, so confidence is fixed at 100.0.
            Degenerate crops retain the same schema with an empty value.

        Raises:
            TypeError: If ``mapped_rois`` is not a dictionary or a non-null ROI
                is not a NumPy array.
        """
        if not isinstance(mapped_rois, dict):
            raise TypeError("mapped_rois must be a dictionary")

        output: dict[str, dict[str, str | float]] = {}
        for field_name, crop in mapped_rois.items():
            if crop is None or (isinstance(crop, np.ndarray) and crop.size == 0):
                extracted_text = ""
            else:
                if not isinstance(crop, np.ndarray):
                    raise TypeError(f"ROI '{field_name}' must be a NumPy array")
                extracted_text = self._recognize(crop)

            output[field_name] = {
                "value": extracted_text,
                "confidence": 100.0,
            }

        return output
