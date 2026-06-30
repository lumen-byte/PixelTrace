"""
PixelTrace - Color Feature Extractor
------------------------------------
Extracts color statistics from an image.
"""

import cv2
import numpy as np


class ColorFeatureExtractor:
    """
    Extract color-related features.
    """

    def extract(self, image: np.ndarray) -> dict:

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)

        # Split RGB channels to calculate covariance/correlation and Laplacian differences
        b, g, r = cv2.split(image)
        
        # Flatten channels to calculate Pearson correlation coefficient
        r_flat = r.ravel().astype(np.float32)
        g_flat = g.ravel().astype(np.float32)
        b_flat = b.ravel().astype(np.float32)
        
        corr_rg = np.corrcoef(r_flat, g_flat)[0, 1]
        corr_gb = np.corrcoef(g_flat, b_flat)[0, 1]
        corr_br = np.corrcoef(b_flat, r_flat)[0, 1]

        # Calculate Laplacian for each channel to measure high-frequency alignment
        l_r = cv2.Laplacian(r, cv2.CV_64F)
        l_g = cv2.Laplacian(g, cv2.CV_64F)
        l_b = cv2.Laplacian(b, cv2.CV_64F)
        
        diff_rg = np.mean(np.abs(l_r - l_g))
        diff_gb = np.mean(np.abs(l_g - l_b))
        diff_br = np.mean(np.abs(l_b - l_r))

        # Calculate higher-order statistical moments of intensity and saturation
        from scipy.stats import skew, kurtosis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_flat = gray.ravel().astype(np.float32)
        s_flat = s.ravel().astype(np.float32)

        gray_skew = skew(gray_flat)
        gray_kurt = kurtosis(gray_flat)
        sat_skew = skew(s_flat)
        sat_kurt = kurtosis(s_flat)

        return {
            "h_mean": round(float(np.mean(h)), 4),
            "s_mean": round(float(np.mean(s)), 4),
            "v_mean": round(float(np.mean(v)), 4),

            "h_std": round(float(np.std(h)), 4),
            "s_std": round(float(np.std(s)), 4),
            "v_std": round(float(np.std(v)), 4),

            "corr_rg": round(float(corr_rg) if not np.isnan(corr_rg) else 0.0, 4),
            "corr_gb": round(float(corr_gb) if not np.isnan(corr_gb) else 0.0, 4),
            "corr_br": round(float(corr_br) if not np.isnan(corr_br) else 0.0, 4),

            "lap_diff_rg": round(float(diff_rg), 4),
            "lap_diff_gb": round(float(diff_gb), 4),
            "lap_diff_br": round(float(diff_br), 4),

            "gray_skew": round(float(gray_skew), 4),
            "gray_kurtosis": round(float(gray_kurt), 4),
            "sat_skew": round(float(sat_skew), 4),
            "sat_kurtosis": round(float(sat_kurt), 4),
        }