"""
PixelTrace - Image Preprocessing Module
---------------------------------------
This module prepares images before feature extraction.
Every feature extractor in PixelTrace uses this preprocessing pipeline.
"""

from pathlib import Path

import cv2
import numpy as np


class ImagePreprocessor:
    """
    Handles all preprocessing operations for PixelTrace.
    """

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from disk.

        Args:
            image_path: Path to image.

        Returns:
            Loaded BGR image.

        Raises:
            FileNotFoundError
            ValueError
        """
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(str(path))

        if image is None:
            raise ValueError("Unable to read image.")

        return image

    def resize(
        self,
        image: np.ndarray,
        width: int = 256
    ) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        """
        h, w = image.shape[:2]

        aspect_ratio = h / w

        height = int(width * aspect_ratio)

        return cv2.resize(
            image,
            (width, height),
            interpolation=cv2.INTER_AREA
        )

    def to_grayscale(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Convert image to grayscale.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_clahe(
        self,
        gray_image: np.ndarray
    ) -> np.ndarray:
        """
        Improve local contrast using CLAHE.
        """
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        return clahe.apply(gray_image)

    def normalize(
        self,
        image: np.ndarray
    ) -> np.ndarray:
        """
        Normalize pixel values to [0,1].
        """
        return image.astype(np.float32) / 255.0

    def preprocess(
        self,
        image_path: str
    ) -> dict:
        """
        Complete preprocessing pipeline.

        Returns
        -------
        Dictionary containing all intermediate images.
        """

        original = self.load_image(image_path)

        resized = self.resize(original)

        gray = self.to_grayscale(resized)

        enhanced = self.apply_clahe(gray)

        normalized = self.normalize(enhanced)

        return {
            "original": original,
            "resized": resized,
            "gray": gray,
            "enhanced": enhanced,
            "normalized": normalized,
        }