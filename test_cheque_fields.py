"""Extract common cheque fields with one shared TrOCR model instance.

This is a local demonstration helper. The normalized regions fit the supplied
sample layout; production processing should obtain equivalent crops from YOLO.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from cv_core.ocr_engine import OCREngine


# Normalized coordinates: (left, top, right, bottom).
FIELD_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "bank_name": (0.025, 0.12, 0.09, 0.20),
    "date": (0.53, 0.12, 0.72, 0.22),
    "payee": (0.17, 0.23, 0.54, 0.38),
    "amount_words": (0.025, 0.35, 0.43, 0.48),
    "amount_number": (0.75, 0.28, 0.90, 0.40),
    "signature": (0.50, 0.59, 0.86, 0.80),
    "routing_number": (0.03, 0.77, 0.28, 0.87),
    "account_number": (0.29, 0.77, 0.54, 0.87),
    "cheque_number": (0.77, 0.10, 0.88, 0.19),
}


def main() -> None:
    """Load one cheque, crop its fields, and print structured OCR output."""
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    args = parser.parse_args()

    image_path = args.image.expanduser().resolve()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise SystemExit(f"Could not decode image: {image_path}")

    height, width = image.shape[:2]
    crops = {}
    for field, (left, top, right, bottom) in FIELD_REGIONS.items():
        x1, y1 = int(left * width), int(top * height)
        x2, y2 = int(right * width), int(bottom * height)
        crops[field] = image[y1:y2, x1:x2]

    print("Loading microsoft/trocr-base-handwritten once...")
    results = OCREngine().extract_all_text(crops)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
