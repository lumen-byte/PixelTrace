"""
PixelTrace - FFT Feature Extractor
----------------------------------
Extracts frequency-domain features from an image.
"""

import cv2
import numpy as np


class FFTFeatureExtractor:
    """
    Extract frequency-domain features using Fast Fourier Transform.
    """

    def extract(self, gray_image: np.ndarray) -> dict:
        """
        Extract FFT features.

        Args:
            gray_image: Preprocessed grayscale image.

        Returns:
            Dictionary containing FFT features.
        """

        # Compute FFT
        fft = np.fft.fft2(gray_image)
        fft_shift = np.fft.fftshift(fft)

        magnitude = np.abs(fft_shift)
        magnitude = np.log1p(magnitude)

        fft_mean = float(np.mean(magnitude))
        fft_std = float(np.std(magnitude))
        fft_max = float(np.max(magnitude))

        features = {
            "fft_mean": round(fft_mean, 4),
            "fft_std": round(fft_std, 4),
            "fft_max": round(fft_max, 4),
        }

        # Size of the FFT image
        h, w = gray_image.shape
        cy, cx = h // 2, w // 2

        # Create coordinate grid of distances (radial) and angles (theta) from center
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((y - cy)**2 + (x - cx)**2)
        theta = np.arctan2(y - cy, x - cx)
        # Fold angle to [0, pi] because magnitude spectrum is conjugate symmetric
        theta_folded = np.abs(theta)

        # 1. Radial bins (concentric rings) - captures different frequency bands (scales)
        num_radial_bins = 10
        max_r = np.sqrt(cy**2 + cx**2)
        radial_edges = np.linspace(0, max_r, num_radial_bins + 1)
        for i in range(num_radial_bins):
            mask = (r >= radial_edges[i]) & (r < radial_edges[i+1])
            if np.any(mask):
                bin_vals = magnitude[mask]
                features[f"fft_radial_mean_{i}"] = round(float(np.mean(bin_vals)), 4)
                features[f"fft_radial_std_{i}"] = round(float(np.std(bin_vals)), 4)
            else:
                features[f"fft_radial_mean_{i}"] = 0.0
                features[f"fft_radial_std_{i}"] = 0.0

        # 2. Azimuthal bins (angular wedges) - captures directional patterns (grid lines)
        num_angular_bins = 8
        angular_edges = np.linspace(0, np.pi, num_angular_bins + 1)
        for i in range(num_angular_bins):
            mask = (theta_folded >= angular_edges[i]) & (theta_folded < angular_edges[i+1])
            if np.any(mask):
                bin_vals = magnitude[mask]
                features[f"fft_angular_mean_{i}"] = round(float(np.mean(bin_vals)), 4)
                features[f"fft_angular_std_{i}"] = round(float(np.std(bin_vals)), 4)
            else:
                features[f"fft_angular_mean_{i}"] = 0.0
                features[f"fft_angular_std_{i}"] = 0.0

        features["fft_image"] = magnitude
        return features