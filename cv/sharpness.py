"""
PixelTrace - Sharpness Feature Extractor
----------------------------------------
Measures image sharpness using the Variance of Laplacian.
"""

import cv2
import numpy as np


class SharpnessFeatureExtractor:
    """
    Extract sharpness-related features.
    """

    def extract(self, gray_image: np.ndarray) -> dict:
        """
        Compute image sharpness.

        Args:
            gray_image: Preprocessed grayscale image.

        Returns:
            Dictionary containing sharpness metrics.
        """

        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

        variance = float(laplacian.var())

        mean = float(np.mean(np.abs(laplacian)))

        return {
            "sharpness_variance": round(variance, 4),
            "sharpness_mean": round(mean, 4),
        }