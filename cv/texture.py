"""
PixelTrace - Texture Feature Extractor
--------------------------------------
Extracts texture features using
1. Local Binary Pattern (LBP)
2. Gray Level Co-occurrence Matrix (GLCM)
"""

import cv2
import numpy as np

from skimage.feature import (
    local_binary_pattern,
    graycomatrix,
    graycoprops,
)


class TextureFeatureExtractor:

    def __init__(self):

        self.radius = 2
        self.points = 16

    def extract(self, gray):

        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        # ---------- LBP ----------

        lbp = local_binary_pattern(
            gray,
            self.points,
            self.radius,
            method="uniform",
        )

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.points + 2,
            range=(0, self.points + 2),
        )

        hist = hist.astype(np.float32)

        hist /= hist.sum() + 1e-8

        # ---------- GLCM ----------

        quantized = (gray // 16).astype(np.uint8)

        glcm = graycomatrix(
            quantized,
            distances=[1],
            angles=[0],
            levels=16,
            symmetric=True,
            normed=True,
        )

        contrast = graycoprops(glcm, "contrast")[0, 0]
        homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
        energy = graycoprops(glcm, "energy")[0, 0]
        correlation = graycoprops(glcm, "correlation")[0, 0]

        features = {

            "texture_mean": round(float(hist.mean()), 4),

            "texture_std": round(float(hist.std()), 4),

            "texture_energy": round(float(energy), 4),

            "glcm_contrast": round(float(contrast), 4),

            "glcm_homogeneity": round(float(homogeneity), 4),

            "glcm_correlation": round(float(correlation), 4),

        }

        for idx, val in enumerate(hist):
            features[f"texture_lbp_bin_{idx}"] = round(float(val), 4)

        return features