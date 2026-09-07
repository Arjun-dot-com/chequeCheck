import cv2
import pytesseract
from pytesseract import Output
import re

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


class OCREngine:
    def __init__(self):
        self.configs = {
            "micr": "--psm 7 -c tessedit_char_whitelist=0123456789",
            "amount_num": "--psm 7 -c tessedit_char_whitelist=0123456789.,",
            "date": "--psm 7 -c tessedit_char_whitelist=0123456789/-",
            "payee": "--psm 7",
            "chq_no": "--psm 7 -c tessedit_char_whitelist=0123456789",
            "bank_name": "--psm 7",
        }

    def _variants(self, image):
        enlarged = cv2.resize(
            image, None, fx=4, fy=4,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

        otsu = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 11
        )

        return [enlarged, gray, otsu, adaptive]

    def _read_text_and_conf(self, image, config):
        best_text = ""
        best_conf = 0.0

        for variant in self._variants(image):
            data = pytesseract.image_to_data(
                variant,
                config=config,
                output_type=Output.DICT
            )

            words = []
            confidences = []

            for text, conf in zip(data["text"], data["conf"]):
                text = text.strip()

                try:
                    conf = float(conf)
                except:
                    conf = -1

                if text and conf >= 0:
                    words.append(text)
                    confidences.append(conf)

            if words:
                text = " ".join(words)
                confidence = sum(confidences) / len(confidences)

                if confidence > best_conf:
                    best_text = text
                    best_conf = confidence

        return {
            "value": best_text,
            "confidence": round(best_conf, 2)
        }

    def _micr_fallback(self, image):
        """
        Fallback specifically for cheque number.
        Looks at the bottom MICR region and searches for
        a six-digit sequence.
        """

        h, w = image.shape[:2]

        # Bottom MICR band
        crop = image[
            int(h * 0.72):int(h * 0.98),
            int(w * 0.05):int(w * 0.95)
        ]

        enlarged = cv2.resize(
            crop, None, fx=5, fy=5,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

        variants = [
            enlarged,
            gray,
            cv2.threshold(
                gray, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )[1]
        ]

        candidates = []

        for variant in variants:
            for psm in [6, 7, 8, 13]:

                text = pytesseract.image_to_string(
                    variant,
                    config=(
                        f"--psm {psm} "
                        "-c tessedit_char_whitelist=0123456789"
                    )
                )

                digits = re.sub(r"\D", "", text)

                # Find six consecutive digits
                matches = re.findall(r"\d{6}", digits)

                for match in matches:
                    candidates.append(match)

        # Prefer repeated candidate
        if candidates:
            counts = {}

            for value in candidates:
                counts[value] = counts.get(value, 0) + 1

            best = max(
                counts,
                key=counts.get
            )

            return {
                "value": best,
                "confidence": 75.0
            }

        return {
            "value": "",
            "confidence": 0.0
        }

    def extract_all_text(self, rois):

        output = {}

        for field, image in rois.items():

            if image is None or image.size == 0:
                output[field] = {
                    "value": "",
                    "confidence": 0.0
                }
                continue

            config = self.configs.get(
                field,
                "--psm 7"
            )

            output[field] = self._read_text_and_conf(
                image,
                config
            )

        # MICR fallback for cheque number
        chq_result = output.get(
            "chq_no",
            {"value": "", "confidence": 0}
        )

        if not chq_result["value"].strip():

            print("Normal cheque-number OCR failed.")
            print("Trying MICR fallback...")

            # We need the original cheque image.
            # This fallback is handled separately by pipeline
            # if original image is supplied.
        
        return output

    def extract_cheque_number_from_micr(self, image):
        return self._micr_fallback(image)