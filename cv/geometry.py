"""
PixelTrace - Geometry Feature Extractor
---------------------------------------
Extracts geometric features from an image.
"""

import cv2
import numpy as np


class GeometryFeatureExtractor:
    """
    Extract geometric properties from an image.
    """

    def extract(self, gray_image: np.ndarray) -> dict:

        edges = cv2.Canny(gray_image, 100, 200)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contour_count = len(contours)

        areas = [cv2.contourArea(c) for c in contours]

        if len(areas) == 0:
            largest_area = 0.0
            mean_area = 0.0
        else:
            largest_area = max(areas)
            mean_area = np.mean(areas)

        h, w = gray_image.shape

        aspect_ratio = w / h

        return {
            "contour_count": contour_count,
            "largest_contour_area": round(float(largest_area), 2),
            "mean_contour_area": round(float(mean_area), 2),
            "aspect_ratio": round(float(aspect_ratio), 4),
        }