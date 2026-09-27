"""Siamese-network checks for signatures and cursive cheque amounts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .snn_model import SNNVerifier


DEFAULT_SNN_MODEL_PATH = (
    Path(__file__).resolve().parent / "models" / "snn" / "best.pt"
)


class FraudChecker:
    """Apply one shared SNN verifier to signatures and cursive word crops.

    The verifier is constructed once and reused by both public checks. This is
    the dual-purpose SNN design: signature and amount comparisons use identical
    learned features while supplying task-specific image pairs.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_SNN_MODEL_PATH,
        distance_threshold: float = 0.5,
        device: str | torch.device | None = None,
    ) -> None:
        """Load the shared SNN verifier for fraud-related comparisons.

        Args:
            model_path: Path to trained Siamese-network weights.
            distance_threshold: Maximum embedding distance considered a match.
            device: Optional explicit PyTorch device. The verifier dynamically
                chooses CUDA or CPU when this argument is omitted.
        """
        self.snn_verifier = SNNVerifier(
            model_path=model_path,
            threshold=distance_threshold,
            device=device,
        )

    @staticmethod
    def _is_degenerate(crop: np.ndarray | None) -> bool:
        """Return whether a crop is missing, invalid, or has no pixels."""
        return not isinstance(crop, np.ndarray) or crop.size == 0

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """Map unit-embedding Euclidean distance from ``[0, 2]`` to ``[1, 0]``."""
        return float(np.clip(1.0 - (distance / 2.0), 0.0, 1.0))

    def check_signature_authenticity(
        self,
        extracted_sign_crop: np.ndarray,
        reference_sign_crop: np.ndarray,
    ) -> dict[str, bool | float]:
        """Compare an extracted signature with the account-holder reference.

        Args:
            extracted_sign_crop: Signature crop extracted from the cheque.
            reference_sign_crop: Stored genuine signature image.

        Returns:
            ``is_signed`` is true only when the SNN considers the pair a match.
            ``similarity_score`` is a normalized score where one is identical
            and zero is maximally distant. Missing crops return a safe
            non-match with zero similarity.
        """
        if self._is_degenerate(extracted_sign_crop) or self._is_degenerate(
            reference_sign_crop
        ):
            return {"is_signed": False, "similarity_score": 0.0}

        verification = self.snn_verifier.verify(
            extracted_sign_crop, reference_sign_crop
        )
        distance = float(verification["distance_score"])
        return {
            "is_signed": bool(verification["is_match"]),
            "similarity_score": self._distance_to_similarity(distance),
        }

    def spot_cursive_amount(
        self,
        extracted_amount_crop: np.ndarray,
        reference_amount_crop: np.ndarray,
    ) -> dict[str, bool]:
        """Compare a cursive amount crop with its expected visual reference.

        Args:
            extracted_amount_crop: Handwritten amount extracted from the cheque.
            reference_amount_crop: Reference rendering or exemplar of the
                expected amount.

        Returns:
            A dictionary containing ``amount_tamper_flag``. It is true when the
            embedding distance exceeds the verifier's configured threshold.
            Missing crops cannot be compared and therefore return false rather
            than asserting tampering without evidence.
        """
        if self._is_degenerate(extracted_amount_crop) or self._is_degenerate(
            reference_amount_crop
        ):
            return {"amount_tamper_flag": False}

        verification = self.snn_verifier.verify(
            extracted_amount_crop, reference_amount_crop
        )
        return {"amount_tamper_flag": not bool(verification["is_match"])}
