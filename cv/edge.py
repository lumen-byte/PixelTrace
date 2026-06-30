"""
PixelTrace - Edge Feature Extractor
-----------------------------------
Extracts edge-based features from an image.
"""

import cv2
import numpy as np


class EdgeFeatureExtractor:
    """
    Extract edge-related features using Canny Edge Detection.
    """

    def __init__(
        self,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def extract(self, gray_image: np.ndarray) -> dict:
        """
        Extract edge features.

        Args:
            gray_image: Preprocessed grayscale image.

        Returns:
            Dictionary of edge features.
        """

        edges = cv2.Canny(
            gray_image,
            self.low_threshold,
            self.high_threshold,
        )

        edge_pixels = np.count_nonzero(edges)

        total_pixels = edges.shape[0] * edges.shape[1]

        edge_density = edge_pixels / total_pixels

        return {
            "edge_map": edges,
            "edge_density": round(edge_density, 4),
            "edge_pixels": int(edge_pixels),
        }