"""
PixelTrace - Structural Feature Extractor (Optimized)
-----------------------------------------------------
Uses CV_32F Sobel + cv2.magnitude/phase (SIMD-accelerated).
"""

import cv2
import numpy as np


class StructuralFeatureExtractor:
    """
    Extract structural features using Sobel gradients.
    """

    def extract(self, gray_image: np.ndarray) -> dict:

        # CV_32F is 2x less memory than CV_64F — faster on weak CPUs
        grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

        # SIMD-accelerated magnitude and phase
        magnitude = cv2.magnitude(grad_x, grad_y)
        direction = cv2.phase(grad_x, grad_y, angleInDegrees=True)
        direction = direction % 180.0

        hist, _ = np.histogram(direction, bins=9, range=(0, 180))
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-6)

        horizontal = float(hist[0] + hist[-1])
        vertical = float(hist[4])
        diagonal = float(hist[2] + hist[6])
        dominant_orientation = int(np.argmax(hist) * 20)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))

        mag_sq = magnitude ** 2

        features = {
            "gradient_mean": round(float(magnitude.mean()), 4),
            "gradient_std": round(float(magnitude.std()), 4),
            "gradient_energy": round(float(mag_sq.sum()), 4),
            "horizontal_ratio": round(horizontal, 4),
            "vertical_ratio": round(vertical, 4),
            "diagonal_ratio": round(diagonal, 4),
            "orientation_entropy": round(entropy, 4),
            "dominant_orientation": dominant_orientation,
        }

        for idx, val in enumerate(hist):
            features[f"hog_bin_{idx}"] = round(float(val), 4)

        return features