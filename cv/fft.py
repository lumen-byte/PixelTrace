"""
PixelTrace - FFT + Moiré Feature Extractor
-------------------------------------------
Extracts frequency-domain and moiré features from a single 2D FFT pass.
Previously these were split across FFT and Moiré extractors, causing
redundant FFT computation.
"""

import cv2
import numpy as np


class FFTFeatureExtractor:
    """
    Extract frequency-domain and moiré forensic features using a single FFT.
    """

    def __init__(self, peak_threshold_sigma: float = 3.0):
        self.peak_threshold_sigma = peak_threshold_sigma

    def extract(self, gray_image: np.ndarray) -> dict:

        img = gray_image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        h, w = img.shape
        cy, cx = h // 2, w // 2

        # Apply Hann window (suppresses spectral leakage — needed for moiré)
        window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        windowed = img * window

        # Single 2D FFT
        fft = np.fft.fft2(windowed)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift)).astype(np.float32)

        # Zero DC
        magnitude[cy, cx] = 0.0

        eps = 1e-12
        total_energy = float(np.sum(magnitude ** 2)) + eps

        fft_mean = float(np.mean(magnitude))
        fft_std = float(np.std(magnitude))
        fft_max = float(np.max(magnitude))

        features = {
            "fft_mean": round(fft_mean, 4),
            "fft_std": round(fft_std, 4),
            "fft_max": round(fft_max, 4),
        }

        # Precompute coordinate grids (shared between radial/angular bins and moiré)
        y_idx, x_idx = np.ogrid[:h, :w]
        r = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
        max_r = min(cy, cx)
        theta = np.arctan2(y_idx - cy, x_idx - cx)
        theta_folded = np.abs(theta)

        # ── FFT radial bins (10 rings) ────────────────────────────────────────
        num_radial_bins = 10
        max_r_diag = np.sqrt(cy ** 2 + cx ** 2)
        radial_edges = np.linspace(0, max_r_diag, num_radial_bins + 1)
        for i in range(num_radial_bins):
            mask = (r >= radial_edges[i]) & (r < radial_edges[i + 1])
            if np.any(mask):
                bin_vals = magnitude[mask]
                features[f"fft_radial_mean_{i}"] = round(float(np.mean(bin_vals)), 4)
                features[f"fft_radial_std_{i}"] = round(float(np.std(bin_vals)), 4)
            else:
                features[f"fft_radial_mean_{i}"] = 0.0
                features[f"fft_radial_std_{i}"] = 0.0

        # ── FFT angular bins (8 wedges) ───────────────────────────────────────
        num_angular_bins = 8
        angular_edges = np.linspace(0, np.pi, num_angular_bins + 1)
        for i in range(num_angular_bins):
            mask = (theta_folded >= angular_edges[i]) & (theta_folded < angular_edges[i + 1])
            if np.any(mask):
                bin_vals = magnitude[mask]
                features[f"fft_angular_mean_{i}"] = round(float(np.mean(bin_vals)), 4)
                features[f"fft_angular_std_{i}"] = round(float(np.std(bin_vals)), 4)
            else:
                features[f"fft_angular_mean_{i}"] = 0.0
                features[f"fft_angular_std_{i}"] = 0.0

        # ── Moiré: Spectral flatness (Wiener entropy) ─────────────────────────
        flat_mag = magnitude.ravel() + eps
        geometric_mean = float(np.exp(np.mean(np.log(flat_mag))))
        arithmetic_mean = float(np.mean(flat_mag))
        spectral_flatness = geometric_mean / (arithmetic_mean + eps)

        # ── Moiré: Peak detection ─────────────────────────────────────────────
        mu = float(np.mean(magnitude))
        sigma = float(np.std(magnitude))
        threshold = mu + self.peak_threshold_sigma * sigma
        peak_mask = magnitude > threshold
        peak_count = int(np.sum(peak_mask))
        peak_energy = float(np.sum((magnitude[peak_mask]) ** 2))
        peak_ratio = peak_energy / total_energy

        # ── Moiré: HF / LF energy ratio ──────────────────────────────────────
        hf_mask = r > 0.70 * max_r
        hf_energy = float(np.sum((magnitude[hf_mask]) ** 2))
        hf_ratio = hf_energy / total_energy

        lf_mask = r < 0.25 * max_r
        lf_mask[cy, cx] = False
        lf_energy = float(np.sum((magnitude[lf_mask]) ** 2))
        lf_ratio = lf_energy / total_energy

        hf_to_lf_ratio = hf_energy / (lf_energy + eps)

        # ── Moiré: Radial ring regularity ─────────────────────────────────────
        num_rings = 16
        ring_edges = np.linspace(0, max_r, num_rings + 1)
        ring_energies = []
        for i in range(num_rings):
            ring_mask = (r >= ring_edges[i]) & (r < ring_edges[i + 1])
            ring_energies.append(float(np.sum((magnitude[ring_mask]) ** 2)))

        ring_arr = np.array(ring_energies, dtype=np.float32) + eps
        ring_cv = float(np.std(ring_arr) / np.mean(ring_arr))

        # ── Moiré: Angular energy dominance ───────────────────────────────────
        num_wedges = 36
        wedge_edges = np.linspace(0, np.pi, num_wedges + 1)
        wedge_energies = []
        for i in range(num_wedges):
            wedge_mask = (theta_folded >= wedge_edges[i]) & (theta_folded < wedge_edges[i + 1])
            wedge_energies.append(float(np.sum((magnitude[wedge_mask]) ** 2)))

        wedge_arr = np.array(wedge_energies, dtype=np.float32) + eps
        wedge_arr_norm = wedge_arr / wedge_arr.sum()
        angular_entropy = float(-np.sum(wedge_arr_norm * np.log2(wedge_arr_norm + eps)))
        angular_max_ratio = float(wedge_arr.max() / wedge_arr.sum())

        # Moiré features
        features["moire_spectral_flatness"] = round(spectral_flatness, 6)
        features["moire_peak_count"] = peak_count
        features["moire_peak_ratio"] = round(peak_ratio, 6)
        features["moire_hf_ratio"] = round(hf_ratio, 6)
        features["moire_lf_ratio"] = round(lf_ratio, 6)
        features["moire_hf_to_lf"] = round(hf_to_lf_ratio, 6)
        features["moire_ring_cv"] = round(ring_cv, 6)
        features["moire_angular_entropy"] = round(angular_entropy, 6)
        features["moire_angular_dominance"] = round(angular_max_ratio, 6)

        return features