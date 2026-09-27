"""Siamese neural network primitives for handwriting comparison.

The shared feature extractor in this module is intentionally task-agnostic: the
same network can be trained and used for signature verification or cursive word
spotting.  Both use cases compare 128-dimensional embeddings with Euclidean
distance; only their reference galleries and calibrated thresholds differ.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork(nn.Module):
    """SigNet-style CNN that creates a 128-dimensional handwriting embedding.

    The two Siamese branches share every parameter because :meth:`forward`
    invokes :meth:`forward_once` for each image.  Inputs must be grayscale
    tensors with shape ``(batch_size, 1, 105, 105)``.
    """

    embedding_size: int = 128
    input_size: tuple[int, int] = (105, 105)

    def __init__(self) -> None:
        """Initialize the convolutional feature extractor and projection head."""
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # A 105x105 input produces a 256x2x2 feature map.
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(1024, self.embedding_size),
        )

    def forward_once(self, image: torch.Tensor) -> torch.Tensor:
        """Encode one batch of handwriting crops into normalized embeddings.

        Args:
            image: Float tensor shaped ``(N, 1, 105, 105)``.

        Returns:
            Tensor shaped ``(N, 128)`` with unit-length feature vectors.

        Raises:
            ValueError: If the input does not have the expected dimensions.
        """
        expected_shape = (1, *self.input_size)
        if image.ndim != 4 or tuple(image.shape[1:]) != expected_shape:
            raise ValueError(
                "Expected input shape (N, 1, 105, 105), "
                f"but received {tuple(image.shape)}."
            )

        features = self.feature_extractor(image)
        embedding = self.embedding_head(features)
        return F.normalize(embedding, p=2, dim=1)

    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode two image batches using the same network parameters.

        Args:
            img1: First batch shaped ``(N, 1, 105, 105)``.
            img2: Second batch shaped ``(N, 1, 105, 105)``.

        Returns:
            A pair of tensors, each shaped ``(N, 128)``.
        """
        return self.forward_once(img1), self.forward_once(img2)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for similar (0) and dissimilar (1) image pairs.

    The implemented objective is exactly
    ``mean((1-label) * D^2 + label * max(0, margin-D)^2)``, where ``D`` is
    Euclidean distance.  A label of zero denotes a genuine/similar pair and a
    label of one denotes a forged/dissimilar pair.
    """

    def __init__(self, margin: float = 1.0) -> None:
        """Initialize the loss with the separation margin.

        Args:
            margin: Minimum desired distance between dissimilar embeddings.

        Raises:
            ValueError: If ``margin`` is not positive.
        """
        super().__init__()
        if margin <= 0.0:
            raise ValueError("margin must be greater than zero")
        self.margin = float(margin)

    def forward(
        self,
        output1: torch.Tensor,
        output2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate mean contrastive loss for a batch of embedding pairs.

        Args:
            output1: First embedding batch shaped ``(N, embedding_size)``.
            output2: Second embedding batch with the same shape as ``output1``.
            label: Pair labels containing zero (similar) or one (dissimilar).

        Returns:
            Scalar tensor containing the mean batch loss.

        Raises:
            ValueError: If embedding shapes or the number of labels differ.
        """
        if output1.shape != output2.shape:
            raise ValueError("output1 and output2 must have identical shapes")
        if output1.ndim != 2:
            raise ValueError("embeddings must have shape (N, embedding_size)")

        labels = label.to(device=output1.device, dtype=output1.dtype).reshape(-1)
        if labels.numel() != output1.shape[0]:
            raise ValueError("one label is required for each embedding pair")

        distance = torch.linalg.vector_norm(output1 - output2, ord=2, dim=1)
        similar_loss = (1.0 - labels) * distance.pow(2)
        dissimilar_loss = labels * torch.clamp(
            self.margin - distance, min=0.0
        ).pow(2)
        return torch.mean(similar_loss + dissimilar_loss)


class SNNVerifier:
    """Load a trained Siamese model once and compare OpenCV image crops.

    Construct this wrapper during application startup and reuse it for every
    request.  Separate instances may use different calibrated thresholds for
    signatures and word spotting while sharing the same trained architecture.
    """

    def __init__(
        self,
        model_path: str | Path,
        threshold: float = 0.5,
        device: str | torch.device | None = None,
    ) -> None:
        """Load model weights and prepare the network for inference.

        Args:
            model_path: Path to a checkpoint containing a raw state dictionary,
                ``state_dict``, or ``model_state_dict``.
            threshold: Maximum Euclidean distance considered a match.
            device: Optional explicit PyTorch device.  CUDA is selected when
                available; otherwise inference defaults to CPU.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
            ValueError: If the threshold or checkpoint format is invalid.
            RuntimeError: If checkpoint parameters do not fit the architecture.
        """
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("threshold must be a finite, non-negative number")

        checkpoint_path = Path(model_path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SNN checkpoint not found: {checkpoint_path}")

        self.threshold = float(threshold)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = SiameseNetwork()
        state_dict = self._load_state_dict(checkpoint_path)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
        """Read a state dictionary from common PyTorch checkpoint layouts."""
        try:
            checkpoint: Any = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            # ``weights_only`` was added after older supported PyTorch releases.
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        candidate: Any = checkpoint
        if isinstance(checkpoint, Mapping):
            for key in ("model_state_dict", "state_dict"):
                if key in checkpoint:
                    candidate = checkpoint[key]
                    break

        if not isinstance(candidate, Mapping) or not candidate:
            raise ValueError("checkpoint does not contain a valid state dictionary")
        if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
            raise ValueError("checkpoint state dictionary contains non-tensor values")

        # DataParallel checkpoints prefix every parameter name with ``module.``.
        return {
            (str(key)[7:] if str(key).startswith("module.") else str(key)): value
            for key, value in candidate.items()
        }

    @staticmethod
    def _prepare_crop(crop: np.ndarray) -> torch.Tensor:
        """Convert an OpenCV crop to a normalized ``(1, 1, 105, 105)`` tensor."""
        if not isinstance(crop, np.ndarray):
            raise TypeError("crop must be a NumPy array")
        if crop.size == 0:
            raise ValueError("crop must not be empty")

        if crop.ndim == 2:
            gray = crop
        elif crop.ndim == 3 and crop.shape[2] == 1:
            gray = crop[:, :, 0]
        elif crop.ndim == 3 and crop.shape[2] == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        elif crop.ndim == 3 and crop.shape[2] == 4:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("crop must be grayscale, BGR, or BGRA")

        resized = cv2.resize(gray, (105, 105), interpolation=cv2.INTER_AREA)
        pixels = resized.astype(np.float32, copy=False)
        if not np.isfinite(pixels).all():
            raise ValueError("crop contains NaN or infinite pixel values")
        if pixels.min() < 0.0:
            raise ValueError("crop contains negative pixel values")
        if pixels.max() > 1.0:
            pixels = pixels / 255.0
        pixels = np.clip(pixels, 0.0, 1.0)

        contiguous = np.ascontiguousarray(pixels)
        return torch.from_numpy(contiguous).unsqueeze(0).unsqueeze(0)

    def verify(self, crop1: np.ndarray, crop2: np.ndarray) -> dict[str, float | bool]:
        """Compare two handwriting crops using Euclidean embedding distance.

        Args:
            crop1: First grayscale, BGR, or BGRA OpenCV image crop.
            crop2: Second grayscale, BGR, or BGRA OpenCV image crop.

        Returns:
            Dictionary with a Python ``float`` under ``distance_score`` and a
            boolean ``is_match`` determined by the configured threshold.
        """
        tensor1 = self._prepare_crop(crop1).to(self.device)
        tensor2 = self._prepare_crop(crop2).to(self.device)

        with torch.no_grad():
            embedding1, embedding2 = self.model(tensor1, tensor2)
            distance = torch.linalg.vector_norm(
                embedding1 - embedding2, ord=2, dim=1
            ).item()

        distance_score = float(distance)
        return {
            "distance_score": distance_score,
            "is_match": bool(distance_score <= self.threshold),
        }
