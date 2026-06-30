"""
PixelTrace - Noise Feature Extractor
------------------------------------
Extracts image noise characteristics for forensic analysis.
"""

import cv2
import numpy as np


class NoiseFeatureExtractor:
    """
    Estimates image noise using residual analysis.
    """

    def extract(self, gray_image: np.ndarray) -> dict:

        # Denoise using Gaussian Blur
        blurred = cv2.GaussianBlur(
            gray_image,
            (5, 5),
            0
        )

        # Residual image
        residual = cv2.absdiff(gray_image, blurred)

        noise_mean = float(np.mean(residual))
        noise_std = float(np.std(residual))
        noise_var = float(np.var(residual))

        high_noise_pixels = np.sum(residual > 20)

        total_pixels = residual.size

        high_noise_ratio = high_noise_pixels / total_pixels

        residual_energy = float(np.sum(residual.astype(np.float32) ** 2))

        return {
            "noise_mean": round(noise_mean, 4),
            "noise_std": round(noise_std, 4),
            "noise_variance": round(noise_var, 4),
            "high_noise_ratio": round(high_noise_ratio, 4),
            "residual_energy": round(residual_energy, 2),
        }