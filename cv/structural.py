"""
PixelTrace - Structural Feature Extractor
-----------------------------------------
Extracts gradient and edge orientation features.
"""

import cv2
import numpy as np


class StructuralFeatureExtractor:
    """
    Extract structural features using Sobel gradients.
    """

    def extract(self, gray_image: np.ndarray) -> dict:

        # Compute Sobel gradients
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

        # Gradient magnitude
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

        # Gradient direction (degrees)
        direction = np.degrees(np.arctan2(grad_y, grad_x))
        direction = (direction + 180) % 180

        # Orientation histogram (Mini-HOG)
        hist, _ = np.histogram(
            direction,
            bins=9,
            range=(0, 180)
        )

        hist = hist.astype(np.float32)
        hist /= (hist.sum() + 1e-6)

        horizontal = hist[0] + hist[-1]
        vertical = hist[4]
        diagonal = hist[2] + hist[6]

        dominant_orientation = int(np.argmax(hist) * 20)

        entropy = -np.sum(hist * np.log2(hist + 1e-10))

        features = {

            "gradient_mean":
                round(float(np.mean(magnitude)), 4),

            "gradient_std":
                round(float(np.std(magnitude)), 4),

            "gradient_energy":
                round(float(np.sum(magnitude ** 2)), 4),

            "horizontal_ratio":
                round(float(horizontal), 4),

            "vertical_ratio":
                round(float(vertical), 4),

            "diagonal_ratio":
                round(float(diagonal), 4),

            "orientation_entropy":
                round(float(entropy), 4),

            "dominant_orientation":
                dominant_orientation
        }

        for idx, val in enumerate(hist):
            features[f"hog_bin_{idx}"] = round(float(val), 4)

        return features