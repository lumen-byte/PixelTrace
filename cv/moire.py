"""
PixelTrace - Moiré / Pixel-Grid Frequency Detector
---------------------------------------------------
Screen recaptures have a physically inevitable signature: the camera sensor
samples the display's regular pixel grid, producing periodic interference
patterns (moiré) visible as discrete peaks in the 2D FFT power spectrum.

Real photos have a smooth, monotonically-decaying 1/f power spectrum with
no sharp periodic peaks. This extractor quantifies the strength, count and
spatial regularity of spectral peaks to distinguish the two classes.

Key features:
- fft_peak_count: number of strong periodic spikes in the spectrum
- fft_peak_ratio: ratio of peak energy to total spectral energy
- fft_high_freq_energy_ratio: high-frequency band energy (screens are HF-rich due to pixel grid)
- fft_spectral_flatness: Wiener entropy — flat for noise, peaky for grids
- fft_radial_peak_regularity: coefficient of variation of dominant ring energies
- fft_angular_dominance: how much energy is concentrated in a single direction
"""

import cv2
import numpy as np


class MoireDetector:
    """
    Detects moiré / pixel-grid periodic frequency patterns using 2D FFT analysis.
    """

    def __init__(self, peak_threshold_sigma: float = 3.0):
        """
        Args:
            peak_threshold_sigma: Number of standard deviations above the mean
                to classify a spectral value as a 'peak'. Higher = more selective.
        """
        self.peak_threshold_sigma = peak_threshold_sigma

    def extract(self, gray_image: np.ndarray) -> dict:
        """
        Extract moiré / frequency-domain forensic features from a grayscale image.

        Args:
            gray_image: Preprocessed uint8 or float grayscale image.

        Returns:
            Dictionary of forensic frequency features.
        """
        # Ensure float32
        if gray_image.dtype != np.float32:
            img = gray_image.astype(np.float32)
        else:
            img = gray_image

        # Normalise to [0,1] if still in uint8 range
        if img.max() > 1.0:
            img = img / 255.0

        h, w = img.shape
        cy, cx = h // 2, w // 2

        # 1. Apply Hann window to suppress spectral leakage at image borders
        window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
        windowed = img * window

        # 2. Compute 2D FFT and shift zero-frequency to centre
        fft = np.fft.fft2(windowed)
        fft_shift = np.fft.fftshift(fft)

        # 3. Log-magnitude spectrum (avoids dynamic range issues)
        magnitude = np.log1p(np.abs(fft_shift)).astype(np.float32)

        # Zero out DC component to focus on AC content
        magnitude[cy, cx] = 0.0

        total_energy = float(np.sum(magnitude ** 2)) + 1e-8

        # ── Feature 1: Spectral flatness (Wiener entropy) ──────────────────────
        # Flat spectrum = noise/natural. Peaked spectrum = periodic grid/screen.
        eps = 1e-12
        flat_mag = magnitude.ravel() + eps
        geometric_mean = float(np.exp(np.mean(np.log(flat_mag))))
        arithmetic_mean = float(np.mean(flat_mag))
        spectral_flatness = geometric_mean / (arithmetic_mean + eps)

        # ── Feature 2: Peak detection ───────────────────────────────────────────
        mu = float(np.mean(magnitude))
        sigma = float(np.std(magnitude))
        threshold = mu + self.peak_threshold_sigma * sigma

        peak_mask = magnitude > threshold
        peak_count = int(np.sum(peak_mask))
        peak_energy = float(np.sum((magnitude[peak_mask]) ** 2))
        peak_ratio = peak_energy / total_energy

        # ── Feature 3: High-frequency band energy ratio ─────────────────────────
        # Screen pixel grid creates energy in the outer (high-freq) annular ring.
        y_idx, x_idx = np.ogrid[:h, :w]
        r = np.sqrt((y_idx - cy) ** 2 + (x_idx - cx) ** 2)
        max_r = min(cy, cx)

        # High-frequency zone: outer 30% of the spectrum
        hf_mask = r > 0.70 * max_r
        hf_energy = float(np.sum((magnitude[hf_mask]) ** 2))
        hf_ratio = hf_energy / total_energy

        # Low-frequency zone: inner 25%
        lf_mask = r < 0.25 * max_r
        lf_mask[cy, cx] = False  # exclude DC
        lf_energy = float(np.sum((magnitude[lf_mask]) ** 2))
        lf_ratio = lf_energy / total_energy

        # HF-to-LF ratio: screens have relatively more HF energy
        hf_to_lf_ratio = hf_energy / (lf_energy + eps)

        # ── Feature 4: Radial ring peak regularity ─────────────────────────────
        # Screens produce peaks at harmonics of the pixel pitch → ring energies
        # show a regular, repeating pattern. Natural images are irregular.
        num_rings = 16
        ring_energies = []
        ring_edges = np.linspace(0, max_r, num_rings + 1)
        for i in range(num_rings):
            ring_mask = (r >= ring_edges[i]) & (r < ring_edges[i + 1])
            e = float(np.sum((magnitude[ring_mask]) ** 2))
            ring_energies.append(e)

        ring_arr = np.array(ring_energies, dtype=np.float32) + eps
        ring_cv = float(np.std(ring_arr) / np.mean(ring_arr))  # coefficient of variation

        # ── Feature 5: Angular energy dominance ────────────────────────────────
        # Screen grids create strong energy in 2 perpendicular directions (horizontal
        # and vertical pixel lines). Measure how concentrated angular energy is.
        num_wedges = 36
        theta = np.arctan2(y_idx - cy, x_idx - cx)
        theta_folded = np.abs(theta)  # [0, pi] because spectrum is symmetric

        wedge_edges = np.linspace(0, np.pi, num_wedges + 1)
        wedge_energies = []
        for i in range(num_wedges):
            wedge_mask = (theta_folded >= wedge_edges[i]) & (theta_folded < wedge_edges[i + 1])
            e = float(np.sum((magnitude[wedge_mask]) ** 2))
            wedge_energies.append(e)

        wedge_arr = np.array(wedge_energies, dtype=np.float32) + eps
        wedge_arr_norm = wedge_arr / wedge_arr.sum()
        # Shannon entropy of angular distribution (lower = more directionally concentrated)
        angular_entropy = float(-np.sum(wedge_arr_norm * np.log2(wedge_arr_norm + eps)))
        angular_max_ratio = float(wedge_arr.max() / wedge_arr.sum())

        return {
            "moire_spectral_flatness": round(float(spectral_flatness), 6),
            "moire_peak_count": peak_count,
            "moire_peak_ratio": round(float(peak_ratio), 6),
            "moire_hf_ratio": round(float(hf_ratio), 6),
            "moire_lf_ratio": round(float(lf_ratio), 6),
            "moire_hf_to_lf": round(float(hf_to_lf_ratio), 6),
            "moire_ring_cv": round(float(ring_cv), 6),
            "moire_angular_entropy": round(float(angular_entropy), 6),
            "moire_angular_dominance": round(float(angular_max_ratio), 6),
        }
