"""
PixelTrace - Reflection Feature Extractor
-----------------------------------------
Detects bright reflection regions.
"""

import cv2
import numpy as np


class ReflectionFeatureExtractor:
    """
    Extract reflection-related features.
    """

    def extract(self, image: np.ndarray) -> dict:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(
            gray,
            240,
            255,
            cv2.THRESH_BINARY
        )

        reflection_pixels = np.count_nonzero(mask)

        total_pixels = mask.shape[0] * mask.shape[1]

        reflection_ratio = reflection_pixels / total_pixels

        return {
            "reflection_ratio": round(reflection_ratio, 6),
            "reflection_pixels": int(reflection_pixels),
        }