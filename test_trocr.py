"""Small command-line smoke test for the ChequeCheck TrOCR engine.

This script intentionally bypasses YOLO and the SNN. For the clearest result,
provide a tightly cropped image containing one handwritten text line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from cv_core.ocr_engine import OCREngine


def parse_args() -> argparse.Namespace:
    """Parse the image path supplied on the command line."""
    parser = argparse.ArgumentParser(
        description="Extract handwritten text from one image using TrOCR."
    )
    parser.add_argument(
        "image",
        type=Path,
        help="Path to a cropped PNG, JPG, or JPEG handwriting image.",
    )
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional pixel coordinates used to crop one text line before OCR.",
    )
    return parser.parse_args()


def main() -> None:
    """Load an image, run TrOCR, and print its text and result schema."""
    args = parse_args()
    image_path = args.image.expanduser().resolve()

    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise SystemExit(f"OpenCV could not decode the image: {image_path}")

    if args.crop is not None:
        x1, y1, x2, y2 = args.crop
        height, width = image.shape[:2]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise SystemExit(
                f"Invalid crop {args.crop}; image dimensions are {width}x{height}."
            )
        image = image[y1:y2, x1:x2]
        print(f"Using crop: ({x1}, {y1}) to ({x2}, {y2})")

    print("Loading microsoft/trocr-base-handwritten...")
    ocr = OCREngine()
    result = ocr.extract_all_text({"image": image})

    print("\nExtracted text:")
    print(result["image"]["value"] or "<no text detected>")
    print("\nFull OCR result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
