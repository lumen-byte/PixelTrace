"""
PixelTrace - Color Feature Extractor
------------------------------------
Extracts color statistics from an image.
No scipy dependency — pure NumPy for skew/kurtosis.
"""

import cv2
import numpy as np


def _skew(arr: np.ndarray) -> float:
    """Biased sample skewness (matches scipy.stats.skew default)."""
    m = arr.mean()
    s = arr.std()
    if s < 1e-10:
        return 0.0
    return float(((arr - m) ** 3).mean() / (s ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    """Excess kurtosis (matches scipy.stats.kurtosis default bias=True)."""
    m = arr.mean()
    s = arr.std()
    if s < 1e-10:
        return 0.0
    return float(((arr - m) ** 4).mean() / (s ** 4) - 3.0)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Fast Pearson correlation without building a full covariance matrix."""
    a_m = a - a.mean()
    b_m = b - b.mean()
    num = np.dot(a_m, b_m)
    den = np.sqrt(np.dot(a_m, a_m) * np.dot(b_m, b_m))
    if den < 1e-10:
        return 0.0
    return float(num / den)


class ColorFeatureExtractor:
    """
    Extract color-related features.
    """

    def extract(self, image: np.ndarray) -> dict:

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        h, s, v = cv2.split(hsv)

        # Split RGB channels
        b, g, r = cv2.split(image)

        # Fast Pearson correlation (no full covariance matrix)
        r_flat = r.ravel().astype(np.float32)
        g_flat = g.ravel().astype(np.float32)
        b_flat = b.ravel().astype(np.float32)

        corr_rg = _pearson(r_flat, g_flat)
        corr_gb = _pearson(g_flat, b_flat)
        corr_br = _pearson(b_flat, r_flat)

        # Laplacian for each channel
        l_r = cv2.Laplacian(r, cv2.CV_64F)
        l_g = cv2.Laplacian(g, cv2.CV_64F)
        l_b = cv2.Laplacian(b, cv2.CV_64F)

        diff_rg = np.mean(np.abs(l_r - l_g))
        diff_gb = np.mean(np.abs(l_g - l_b))
        diff_br = np.mean(np.abs(l_b - l_r))

        # Higher-order moments — pure NumPy (no scipy)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_flat = gray.ravel().astype(np.float32)
        s_flat = s.ravel().astype(np.float32)

        gray_skew = _skew(gray_flat)
        gray_kurt = _kurtosis(gray_flat)
        sat_skew = _skew(s_flat)
        sat_kurt = _kurtosis(s_flat)

        return {
            "h_mean": round(float(np.mean(h)), 4),
            "s_mean": round(float(np.mean(s)), 4),
            "v_mean": round(float(np.mean(v)), 4),

            "h_std": round(float(np.std(h)), 4),
            "s_std": round(float(np.std(s)), 4),
            "v_std": round(float(np.std(v)), 4),

            "corr_rg": round(corr_rg, 4),
            "corr_gb": round(corr_gb, 4),
            "corr_br": round(corr_br, 4),

            "lap_diff_rg": round(float(diff_rg), 4),
            "lap_diff_gb": round(float(diff_gb), 4),
            "lap_diff_br": round(float(diff_br), 4),

            "gray_skew": round(gray_skew, 4),
            "gray_kurtosis": round(gray_kurt, 4),
            "sat_skew": round(sat_skew, 4),
            "sat_kurtosis": round(sat_kurt, 4),
        }