"""
PixelTrace - Chromatic Aberration Feature Extractor
----------------------------------------------------
Real camera lenses exhibit chromatic aberration (CA): colour fringing at
high-contrast edges caused by wavelength-dependent refraction. The effect is
smooth, radially symmetric, and proportional to distance from the image centre.

Screen recaptures show CA artifacts that are DIFFERENT in two important ways:
1. The display adds its own colour subpixel layout (R-G-B stripe or PenTile)
   which produces a repeating discrete colour pattern at pixel edges.
2. The camera's CA overlays on the screen's CA, producing anomalous fringing
   patterns that do not match the smooth radial model of a single lens system.

Key features:
- ca_rg_shift: mean pixel-level offset between R and G channel edge maps
- ca_gb_shift: mean pixel-level offset between G and B channel edge maps
- ca_edge_color_variance: variance of colour at detected edges (high = CA)
- ca_fringe_ratio: proportion of edges with strong colour fringing
- subpixel_grid_score: evidence of display subpixel grid structure
"""

import cv2
import numpy as np


class ChromaticAberrationExtractor:
    """
    Measures chromatic aberration and display subpixel artefacts.
    """

    def __init__(self, edge_threshold: float = 30.0):
        self.edge_threshold = edge_threshold

    def extract(self, image: np.ndarray) -> dict:
        """
        Args:
            image: BGR uint8 image (resized, not grayscale).

        Returns:
            Dictionary of chromatic aberration features.
        """
        b, g, r = cv2.split(image)

        # ── 1. Edge maps for each channel ──────────────────────────────────────
        edges_r = cv2.Canny(r, self.edge_threshold, self.edge_threshold * 2)
        edges_g = cv2.Canny(g, self.edge_threshold, self.edge_threshold * 2)
        edges_b = cv2.Canny(b, self.edge_threshold, self.edge_threshold * 2)

        eps = 1e-8

        # ── 2. Channel edge misalignment (CA shift proxy) ──────────────────────
        # Dilate each edge map slightly to tolerate 1-2px shift, then measure
        # how much R and B edges land in G edge regions vs outside them.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edges_g_dilated = cv2.dilate(edges_g, kernel)

        rg_overlap = float(np.sum((edges_r > 0) & (edges_g_dilated > 0)))
        gb_overlap = float(np.sum((edges_b > 0) & (edges_g_dilated > 0)))
        total_g_edge = float(np.sum(edges_g > 0)) + eps

        # Lower overlap → channels are more misaligned (higher CA)
        ca_rg_alignment = rg_overlap / total_g_edge
        ca_gb_alignment = gb_overlap / total_g_edge
        ca_mean_alignment = (ca_rg_alignment + ca_gb_alignment) / 2.0

        # ── 3. Colour variance at edges (fringing) ─────────────────────────────
        # At edges of a real lens image, colour channels differ due to CA.
        # At screen recapture edges, the colour pattern is more chaotic (two CAs).
        combined_edges = (edges_r > 0) | (edges_g > 0) | (edges_b > 0)
        if np.sum(combined_edges) > 0:
            r_at_edge = r[combined_edges].astype(np.float32)
            g_at_edge = g[combined_edges].astype(np.float32)
            b_at_edge = b[combined_edges].astype(np.float32)
            edge_rg_diff = float(np.mean(np.abs(r_at_edge - g_at_edge)))
            edge_gb_diff = float(np.mean(np.abs(g_at_edge - b_at_edge)))
            edge_color_var = float(np.var(r_at_edge - g_at_edge))
        else:
            edge_rg_diff = 0.0
            edge_gb_diff = 0.0
            edge_color_var = 0.0

        # ── 4. Subpixel grid score ─────────────────────────────────────────────
        # Screen pixel grids create horizontal/vertical colour banding at
        # 1-3px periodicity. Detect via autocorrelation of R-G difference map.
        rg_diff_map = r.astype(np.int16) - g.astype(np.int16)
        rg_diff_norm = rg_diff_map.astype(np.float32)

        # Horizontal autocorrelation at lag 1, 2, 3 (pixel-pitch period)
        h_autocorr_scores = []
        for lag in [1, 2, 3]:
            a = rg_diff_norm[:, :-lag].ravel()
            b_arr = rg_diff_norm[:, lag:].ravel()
            denom = (np.std(a) * np.std(b_arr) * len(a))
            if denom > eps:
                h_autocorr_scores.append(float(np.dot(a, b_arr) / denom))
            else:
                h_autocorr_scores.append(0.0)

        subpixel_h_score = float(np.max(h_autocorr_scores))

        # Vertical autocorrelation at lag 1, 2, 3
        v_autocorr_scores = []
        for lag in [1, 2, 3]:
            a = rg_diff_norm[:-lag, :].ravel()
            b_arr = rg_diff_norm[lag:, :].ravel()
            denom = np.std(a) * np.std(b_arr) * len(a)
            if denom > eps:
                v_autocorr_scores.append(float(np.dot(a, b_arr) / denom))
            else:
                v_autocorr_scores.append(0.0)

        subpixel_v_score = float(np.max(v_autocorr_scores))

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
