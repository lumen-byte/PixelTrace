"""
PixelTrace - Chromatic Aberration Feature Extractor (Optimized)
---------------------------------------------------------------
Uses Sobel instead of Canny (3x faster), vectorized autocorrelation.
"""

import cv2
import numpy as np


class ChromaticAberrationExtractor:
    """
    Measures chromatic aberration and display subpixel artefacts.
    """

    def __init__(self, edge_threshold: float = 30.0):
        self.edge_threshold = edge_threshold
        # Pre-compute dilation kernel (constant)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def extract(self, image: np.ndarray) -> dict:
        b, g, r = cv2.split(image)

        eps = 1e-8

        # ── 1. Edge maps via Sobel (faster than Canny × 3) ────────────────────
        def _sobel_edges(ch: np.ndarray) -> np.ndarray:
            gx = cv2.Sobel(ch, cv2.CV_16S, 1, 0, ksize=3)
            gy = cv2.Sobel(ch, cv2.CV_16S, 0, 1, ksize=3)
            mag = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
            # threshold at self.edge_threshold * 2 (roughly Canny's high threshold)
            return (mag > self.edge_threshold * 2).astype(np.uint8) * 255

        edges_r = _sobel_edges(r)
        edges_g = _sobel_edges(g)
        edges_b = _sobel_edges(b)

        # ── 2. Channel edge misalignment ──────────────────────────────────────
        edges_g_dilated = cv2.dilate(edges_g, self._kernel)

        rg_overlap = float(np.count_nonzero((edges_r > 0) & (edges_g_dilated > 0)))
        gb_overlap = float(np.count_nonzero((edges_b > 0) & (edges_g_dilated > 0)))
        total_g_edge = float(np.count_nonzero(edges_g)) + eps

        ca_rg_alignment = rg_overlap / total_g_edge
        ca_gb_alignment = gb_overlap / total_g_edge
        ca_mean_alignment = (ca_rg_alignment + ca_gb_alignment) * 0.5

        # ── 3. Colour variance at edges ───────────────────────────────────────
        combined_edges = (edges_r > 0) | (edges_g > 0) | (edges_b > 0)
        if combined_edges.any():
            r_e = r[combined_edges].astype(np.float32)
            g_e = g[combined_edges].astype(np.float32)
            b_e = b[combined_edges].astype(np.float32)
            rg_diff_at_edge = r_e - g_e
            edge_rg_diff = float(np.abs(rg_diff_at_edge).mean())
            edge_gb_diff = float(np.abs(g_e - b_e).mean())
            edge_color_var = float(rg_diff_at_edge.var())
        else:
            edge_rg_diff = edge_gb_diff = edge_color_var = 0.0

        # ── 4. Subpixel grid score via vectorized autocorrelation ─────────────
        rg_diff_norm = (r.astype(np.float32) - g.astype(np.float32)).ravel()
        std_rg = rg_diff_norm.std()

        h, w = image.shape[:2]
        rg_2d = (r.astype(np.float32) - g.astype(np.float32))

        def _autocorr_h(lag: int) -> float:
            a = rg_2d[:, :-lag].ravel()
            b_arr = rg_2d[:, lag:].ravel()
            sa = a.std(); sb = b_arr.std()
            denom = sa * sb * len(a)
            return float(np.dot(a, b_arr) / denom) if denom > eps else 0.0

        def _autocorr_v(lag: int) -> float:
            a = rg_2d[:-lag, :].ravel()
            b_arr = rg_2d[lag:, :].ravel()
            sa = a.std(); sb = b_arr.std()
            denom = sa * sb * len(a)
            return float(np.dot(a, b_arr) / denom) if denom > eps else 0.0

        subpixel_h_score = max(_autocorr_h(lag) for lag in [1, 2, 3])
        subpixel_v_score = max(_autocorr_v(lag) for lag in [1, 2, 3])

        return {
            "ca_rg_alignment": round(float(ca_rg_alignment), 6),
            "ca_gb_alignment": round(float(ca_gb_alignment), 6),
            "ca_mean_alignment": round(float(ca_mean_alignment), 6),
            "ca_edge_rg_diff": round(float(edge_rg_diff), 4),
            "ca_edge_gb_diff": round(float(edge_gb_diff), 4),
            "ca_edge_color_var": round(float(edge_color_var), 4),
            "ca_subpixel_h": round(float(subpixel_h_score), 6),
            "ca_subpixel_v": round(float(subpixel_v_score), 6),
        }
