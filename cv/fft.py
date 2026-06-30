"""
PixelTrace - FFT + Moiré Feature Extractor (Optimized)
-------------------------------------------------------
Single-pass FFT with cv2.dft (hardware-accelerated) and fully
vectorized radial/angular binning via np.digitize.
"""

import cv2
import numpy as np


class FFTFeatureExtractor:
    """
    Extract frequency-domain and moiré forensic features using a single FFT.
    Uses cv2.dft (2x faster than np.fft.fft2) and vectorized binning.
    """

    def __init__(self, peak_threshold_sigma: float = 3.0):
        self.peak_threshold_sigma = peak_threshold_sigma
        # Cache for coordinate grids (keyed by image shape)
        self._cache_shape = None
        self._cache = {}

    def _get_grids(self, h: int, w: int):
        """Cache coordinate grids — they only depend on image dimensions."""
        if self._cache_shape == (h, w):
            return self._cache

        cy, cx = h // 2, w // 2
        max_r = min(cy, cx)
        max_r_diag = np.sqrt(cy ** 2 + cx ** 2)

        # Build coordinate arrays (full 2D)
        y_arr = np.arange(h, dtype=np.float32).reshape(-1, 1) - cy
        x_arr = np.arange(w, dtype=np.float32).reshape(1, -1) - cx
        r = np.sqrt(y_arr ** 2 + x_arr ** 2)
        theta_folded = np.abs(np.arctan2(y_arr, x_arr))

        # Hann window
        window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)

        # Pre-digitize radial bins (10 rings)
        radial_edges = np.linspace(0, max_r_diag, 11)
        radial_bin = np.digitize(r, radial_edges) - 1  # 0..9
        radial_bin = np.clip(radial_bin, 0, 9)

        # Pre-digitize angular bins (8 wedges)
        angular_edges = np.linspace(0, np.pi, 9)
        angular_bin = np.digitize(theta_folded, angular_edges) - 1
        angular_bin = np.clip(angular_bin, 0, 7)

        # Moiré masks
        hf_mask = r > 0.70 * max_r
        lf_mask = r < 0.25 * max_r
        lf_mask[cy, cx] = False

        # Moiré ring bins (16 rings)
        ring_edges = np.linspace(0, max_r, 17)
        ring_bin = np.digitize(r, ring_edges) - 1
        ring_bin = np.clip(ring_bin, 0, 15)

        # Moiré wedge bins (36 wedges)
        wedge_edges = np.linspace(0, np.pi, 37)
        wedge_bin = np.digitize(theta_folded, wedge_edges) - 1
        wedge_bin = np.clip(wedge_bin, 0, 35)

        self._cache = {
            "cy": cy, "cx": cx, "max_r": max_r,
            "window": window,
            "radial_bin": radial_bin, "angular_bin": angular_bin,
            "hf_mask": hf_mask, "lf_mask": lf_mask,
            "ring_bin": ring_bin, "wedge_bin": wedge_bin,
        }
        self._cache_shape = (h, w)
        return self._cache

    def extract(self, gray_image: np.ndarray) -> dict:

        img = gray_image.astype(np.float32)
        if img.max() > 1.0:
            img *= (1.0 / 255.0)

        h, w = img.shape
        g = self._get_grids(h, w)
        cy, cx = g["cy"], g["cx"]

        # Windowed FFT via cv2.dft (hardware-accelerated)
        windowed = img * g["window"]
        dft = cv2.dft(windowed, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft, axes=[0, 1])
        magnitude = np.log1p(
            cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
        ).astype(np.float32)

        # Zero DC
        magnitude[cy, cx] = 0.0

        eps = 1e-12
        mag_sq = magnitude ** 2
        total_energy = float(mag_sq.sum()) + eps

        features = {
            "fft_mean": round(float(magnitude.mean()), 4),
            "fft_std": round(float(magnitude.std()), 4),
            "fft_max": round(float(magnitude.max()), 4),
        }

        # ── Vectorized radial bins (10) ──────────────────────────────────────
        rb = g["radial_bin"]
        for i in range(10):
            mask = rb == i
            if mask.any():
                vals = magnitude[mask]
                features[f"fft_radial_mean_{i}"] = round(float(vals.mean()), 4)
                features[f"fft_radial_std_{i}"] = round(float(vals.std()), 4)
            else:
                features[f"fft_radial_mean_{i}"] = 0.0
                features[f"fft_radial_std_{i}"] = 0.0

        # ── Vectorized angular bins (8) ──────────────────────────────────────
        ab = g["angular_bin"]
        for i in range(8):
            mask = ab == i
            if mask.any():
                vals = magnitude[mask]
                features[f"fft_angular_mean_{i}"] = round(float(vals.mean()), 4)
                features[f"fft_angular_std_{i}"] = round(float(vals.std()), 4)
            else:
                features[f"fft_angular_mean_{i}"] = 0.0
                features[f"fft_angular_std_{i}"] = 0.0

        # ── Moiré: Spectral flatness ─────────────────────────────────────────
        flat_mag = magnitude.ravel() + eps
        geometric_mean = float(np.exp(np.mean(np.log(flat_mag))))
        arithmetic_mean = float(np.mean(flat_mag))
        spectral_flatness = geometric_mean / (arithmetic_mean + eps)

        # ── Moiré: Peak detection ────────────────────────────────────────────
        mu = float(magnitude.mean())
        sigma = float(magnitude.std())
        threshold = mu + self.peak_threshold_sigma * sigma
        peak_mask = magnitude > threshold
        peak_count = int(peak_mask.sum())
        peak_energy = float(mag_sq[peak_mask].sum())
        peak_ratio = peak_energy / total_energy

        # ── Moiré: HF / LF energy ───────────────────────────────────────────
        hf_energy = float(mag_sq[g["hf_mask"]].sum())
        hf_ratio = hf_energy / total_energy
        lf_energy = float(mag_sq[g["lf_mask"]].sum())
        lf_ratio = lf_energy / total_energy
        hf_to_lf_ratio = hf_energy / (lf_energy + eps)

        # ── Moiré: Ring regularity (vectorized) ─────────────────────────────
        ring_energies = np.zeros(16, dtype=np.float64)
        np.add.at(ring_energies, g["ring_bin"].ravel(), mag_sq.ravel())
        ring_energies += eps
        ring_cv = float(ring_energies.std() / ring_energies.mean())

        # ── Moiré: Angular dominance (vectorized) ───────────────────────────
        wedge_energies = np.zeros(36, dtype=np.float64)
        np.add.at(wedge_energies, g["wedge_bin"].ravel(), mag_sq.ravel())
        wedge_energies += eps
        wedge_norm = wedge_energies / wedge_energies.sum()
        angular_entropy = float(-np.sum(wedge_norm * np.log2(wedge_norm + eps)))
        angular_max_ratio = float(wedge_energies.max() / wedge_energies.sum())

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