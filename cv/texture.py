"""
PixelTrace - Texture Feature Extractor (Optimized)
--------------------------------------------------
LBP via integer pixel shifts (r=1, 8-neighbours) — fastest pure-NumPy path.
GLCM via np.bincount — avoids np.add.at scatter overhead.
"""

import cv2
import numpy as np


# ── Pre-computed GLCM weight matrices (constant, computed once at import) ──────
_GLCM_LEVELS = 8
_GLCM_IDX = np.arange(_GLCM_LEVELS, dtype=np.float64)
_I_MAT, _J_MAT = np.meshgrid(_GLCM_IDX, _GLCM_IDX, indexing='ij')
_DIFF_SQ = (_I_MAT - _J_MAT) ** 2                    # contrast weights
_HOMO_W = 1.0 / (1.0 + np.abs(_I_MAT - _J_MAT))     # homogeneity weights


def _lbp_fast(gray: np.ndarray, radius: int = 2, points: int = 16) -> np.ndarray:
    """
    16-point LBP at radius=2 via integer pixel shifts + lookup-table popcount.
    Same uniform encoding as skimage (≤2 transitions → popcount, else → points+1).
    Returns flattened 1-D LBP map of interior pixels.
    """
    import math
    h, w = gray.shape
    g = gray.astype(np.int32)

    # Precompute integer neighbour offsets once
    offsets = []
    for i in range(points):
        angle = 2 * math.pi * i / points
        dy = int(round(-radius * math.cos(angle)))
        dx = int(round(radius * math.sin(angle)))
        offsets.append((dy, dx))

    # Valid interior region (avoids border clamping)
    top = radius; left = radius
    bot = h - radius; right = w - radius
    center = g[top:bot, left:right]

    # Accumulate LBP as 16-bit integer (points=16 bits)
    lbp = np.zeros(center.shape, dtype=np.uint32)
    for bit, (dy, dx) in enumerate(offsets):
        ny, nx = top + dy, left + dx
        neighbour = g[ny:ny + center.shape[0], nx:nx + center.shape[1]]
        lbp |= ((neighbour >= center).astype(np.uint32) << bit)

    lbp = lbp.astype(np.uint32)

    # Circular shift right by 1 (popcount of transitions)
    lbp_shifted = ((lbp >> 1) | ((lbp & 1) << (points - 1))).astype(np.uint32)
    diff = (lbp ^ lbp_shifted).astype(np.uint32)

    # Popcount via 8-bit LUT (split 32-bit int into bytes)
    pc_lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    def _popcount32(arr):
        return (pc_lut[arr & 0xFF].astype(np.uint32) +
                pc_lut[(arr >> 8) & 0xFF].astype(np.uint32) +
                pc_lut[(arr >> 16) & 0xFF].astype(np.uint32) +
                pc_lut[(arr >> 24) & 0xFF].astype(np.uint32))

    transitions = _popcount32(diff)
    n_ones = _popcount32(lbp)

    # Uniform → n_ones (0..points), non-uniform → points+1
    result = np.where(transitions <= 2, n_ones, points + 1).astype(np.uint8)
    return result.ravel()


def _glcm_features(gray: np.ndarray) -> dict:
    """
    GLCM via np.bincount — 4x faster than np.add.at.
    8-level quantization for a 8×8 = 64-element matrix.
    """
    levels = _GLCM_LEVELS
    step = 256 // levels
    q = np.clip(gray.astype(np.int32) // step, 0, levels - 1)

    left = q[:, :-1].ravel()
    right = q[:, 1:].ravel()
    idx = left * levels + right

    glcm_flat = np.bincount(idx, minlength=levels * levels).astype(np.float64)
    glcm = glcm_flat.reshape(levels, levels)
    glcm = glcm + glcm.T  # symmetric
    total = glcm.sum()
    if total > 0:
        glcm /= total

    contrast = float(np.sum(glcm * _DIFF_SQ))
    homogeneity = float(np.sum(glcm * _HOMO_W))
    energy = float(np.sum(glcm ** 2))

    mu_i = float(np.sum(_I_MAT * glcm))
    mu_j = float(np.sum(_J_MAT * glcm))
    sig_i = float(np.sqrt(np.sum(glcm * (_I_MAT - mu_i) ** 2)))
    sig_j = float(np.sqrt(np.sum(glcm * (_J_MAT - mu_j) ** 2)))
    denom = sig_i * sig_j
    if denom > 1e-10:
        correlation = float(np.sum(glcm * (_I_MAT - mu_i) * (_J_MAT - mu_j)) / denom)
    else:
        correlation = 0.0

    return {
        "contrast": contrast,
        "homogeneity": homogeneity,
        "energy": energy,
        "correlation": correlation,
    }


class TextureFeatureExtractor:

    _RADIUS = 2
    _POINTS = 16
    _N_BINS = 18  # points + 2 = 18 (matches original skimage uniform LBP bin count)

    def __init__(self):
        pass  # stateless — all heavy state is module-level constants

    def extract(self, gray):

        if len(gray.shape) == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        # ---------- LBP (fast integer-offset, r=2, p=16) ----------
        lbp_flat = _lbp_fast(gray, self._RADIUS, self._POINTS)  # values 0..17

        # np.bincount is ~5x faster than np.histogram for integer arrays
        hist = np.bincount(lbp_flat.astype(np.int32), minlength=self._N_BINS)
        hist = hist[:self._N_BINS].astype(np.float32)
        hist /= hist.sum() + 1e-8

        # ---------- GLCM (8-level, cached weights) ----------
        glcm_props = _glcm_features(gray)

        features = {
            "texture_mean": round(float(hist.mean()), 4),
            "texture_std": round(float(hist.std()), 4),
            "texture_energy": round(glcm_props["energy"], 4),
            "glcm_contrast": round(glcm_props["contrast"], 4),
            "glcm_homogeneity": round(glcm_props["homogeneity"], 4),
            "glcm_correlation": round(glcm_props["correlation"], 4),
        }

        for idx in range(self._N_BINS):
            features[f"texture_lbp_bin_{idx}"] = round(float(hist[idx]), 4)

        return features