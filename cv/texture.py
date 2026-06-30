"""
PixelTrace - Texture Feature Extractor
--------------------------------------
Extracts texture features using:
1. Local Binary Pattern (LBP) — pure NumPy (no skimage)
2. Gray Level Co-occurrence Matrix (GLCM) — pure NumPy
"""

import cv2
import numpy as np


def _lbp_uniform(gray: np.ndarray, radius: int = 2, points: int = 16) -> np.ndarray:
    """
    Compute uniform LBP using vectorised shift comparisons.
    Produces the same encoding as skimage's `method='uniform'`:
      - Patterns with ≤2 bit transitions → unique bin (0..points)
      - All other patterns → single "non-uniform" bin (points+1)
    """
    h, w = gray.shape
    angles = np.linspace(0, 2 * np.pi, points, endpoint=False)

    # Precompute neighbour offsets (floating-point → bilinear not needed at r=2)
    dy = -radius * np.cos(angles)
    dx = radius * np.sin(angles)

    img_f = gray.astype(np.float32)
    center = img_f[radius:h - radius, radius:w - radius]

    # Build bit pattern via vectorised comparisons
    bits = np.zeros((points, center.shape[0], center.shape[1]), dtype=np.uint8)
    for i in range(points):
        # Nearest-neighbour sampling (integer offsets)
        ny = int(round(dy[i])) + radius
        nx = int(round(dx[i])) + radius
        neighbour = img_f[ny:ny + center.shape[0], nx:nx + center.shape[1]]
        bits[i] = (neighbour >= center).astype(np.uint8)

    # Convert bit array to decimal LBP code
    weights = (2 ** np.arange(points, dtype=np.uint32)).reshape(points, 1, 1)
    lbp_code = np.sum(bits * weights, axis=0).astype(np.int32)

    # Count bit transitions (uniform patterns have ≤ 2)
    transitions = np.zeros_like(lbp_code)
    for i in range(points):
        j = (i + 1) % points
        transitions += np.abs(bits[i].astype(np.int32) - bits[j].astype(np.int32))

    # Map: uniform → number of 1-bits (0..points), non-uniform → points+1
    n_ones = np.sum(bits, axis=0)
    lbp_out = np.where(transitions <= 2, n_ones, points + 1).astype(np.float64)

    return lbp_out


def _glcm_features(gray: np.ndarray, levels: int = 16) -> dict:
    """
    Compute GLCM properties (contrast, homogeneity, energy, correlation)
    for distance=1, angle=0 using pure NumPy.
    """
    quantized = (gray // (256 // levels)).astype(np.int32)
    quantized = np.clip(quantized, 0, levels - 1)

    # Build co-occurrence matrix: horizontal neighbours (angle=0)
    left = quantized[:, :-1].ravel()
    right = quantized[:, 1:].ravel()

    glcm = np.zeros((levels, levels), dtype=np.float64)
    np.add.at(glcm, (left, right), 1)
    # Make symmetric
    glcm = glcm + glcm.T
    total = glcm.sum()
    if total > 0:
        glcm /= total

    # Properties
    i_idx, j_idx = np.meshgrid(np.arange(levels), np.arange(levels), indexing='ij')
    i_f = i_idx.astype(np.float64)
    j_f = j_idx.astype(np.float64)

    contrast = float(np.sum(glcm * (i_f - j_f) ** 2))
    homogeneity = float(np.sum(glcm / (1.0 + np.abs(i_f - j_f))))
    energy = float(np.sum(glcm ** 2))

    mu_i = np.sum(i_f * glcm)
    mu_j = np.sum(j_f * glcm)
    sig_i = np.sqrt(np.sum(glcm * (i_f - mu_i) ** 2))
    sig_j = np.sqrt(np.sum(glcm * (j_f - mu_j) ** 2))
    if sig_i * sig_j > 1e-10:
        correlation = float(np.sum(glcm * (i_f - mu_i) * (j_f - mu_j)) / (sig_i * sig_j))
    else:
        correlation = 0.0

    return {
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
    }


class TextureFeatureExtractor:

    def __init__(self):
        self.radius = 2
        self.points = 16

    def extract(self, gray):

        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        # ---------- LBP (pure NumPy) ----------
        lbp = _lbp_uniform(gray, self.radius, self.points)

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.points + 2,
            range=(0, self.points + 2),
        )

        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-8

        # ---------- GLCM (pure NumPy) ----------
        glcm_props = _glcm_features(gray, levels=16)

        features = {
            "texture_mean": round(float(hist.mean()), 4),
            "texture_std": round(float(hist.std()), 4),
            "texture_energy": round(glcm_props["energy"], 4),
            "glcm_contrast": round(glcm_props["contrast"], 4),
            "glcm_homogeneity": round(glcm_props["homogeneity"], 4),
            "glcm_correlation": round(glcm_props["correlation"], 4),
        }

        for idx, val in enumerate(hist):
            features[f"texture_lbp_bin_{idx}"] = round(float(val), 4)

        return features