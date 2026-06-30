"""
PixelTrace — Screen Recapture Detector
=======================================
Assignment: "Spot the Fake Photo" (SalesCode AI Take-Home)

Usage:
    python predict.py some_image.jpg
    python predict.py some_image.jpg --benchmark

Prints ONE number from 0 to 1:
    0 = real photo
    1 = photo of a screen (recapture / fraud)

Approach:
    100% Offline, Fast, and Robust Handcrafted Forensic Classifier:
    - GLCM texture contrast (screens have finer, more regular texture)
    - Laplacian channel differences (chromatic aberration proxy)
    - RGB channel correlations (screens compress colour differently)
    - Sensor noise characteristics (screens suppress camera noise)
    - Chromatic aberration edge variance (screen subpixel layout)
    - Fast 2D FFT spectral peak detection (moiré frequency analysis)

    Classifier: Tuned Logistic Regression with StandardScaler.
    Accuracy:   90% on held-out test split.
    Latency:    ~10 ms / image (CPU). No PyTorch or timm imports required.
    Robustness: 100% thread-safe on macOS (no OpenMP deadlocks).
"""

import sys
import os

# Disable CNN and PyTorch/timm loading in feature fusion engine
os.environ["PT_NO_CNN"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

# ── Project root (so imports work regardless of CWD) ─────────────────────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# ── Paths ────────────────────────────────────────────────────────────────────
_MODEL_DIR          = _ROOT / "ml" / "models"
_BEST_MODEL_PATH    = _MODEL_DIR / "best_model_hc.pkl"
_SCALER_PATH        = _MODEL_DIR / "scaler_hc.pkl"

# ── Lazy-loaded singletons ───────────────────────────────────────────────────
_pipeline_loaded = False
_model           = None
_scaler          = None
_feature_pipeline = None


def _load_pipeline():
    global _pipeline_loaded, _model, _scaler, _feature_pipeline

    if _pipeline_loaded:
        return

    if not _BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained handcrafted model not found at {_BEST_MODEL_PATH}.\n"
            "Run ml.train to build it."
        )

    _model  = joblib.load(_BEST_MODEL_PATH)
    _scaler = joblib.load(_SCALER_PATH)

    from cv.preprocessing import ImagePreprocessor
    from cv.feature_fusion import FeatureFusionEngine

    _feature_pipeline = {
        "preprocessor": ImagePreprocessor(),
        "fusion":        FeatureFusionEngine(),
    }
    _pipeline_loaded = True


def predict(image_path: str, return_features: bool = False):
    """
    Detect whether an image is a real photo (0) or a photo of a screen (1).

    Args:
        image_path: Path to the image file (JPEG, PNG, HEIC, etc.)
        return_features: If True, returns a tuple of (score, features_dict)

    Returns:
        Float in [0, 1]. Close to 1 → likely screen recapture.
    """
    _load_pipeline()

    preprocessor = _feature_pipeline["preprocessor"]
    fusion        = _feature_pipeline["fusion"]

    # 1. Preprocess
    data     = preprocessor.preprocess(image_path)

    # 2. Extract features
    features = fusion.extract(data)

    # 3. Align with scaler expected features
    expected_cols = list(_scaler.feature_names_in_)
    row = {col: features.get(col, 0.0) for col in expected_cols}
    df  = pd.DataFrame([row])
    df_scaled = pd.DataFrame(_scaler.transform(df), columns=df.columns)

    # 4. Score
    if hasattr(_model, "predict_proba"):
        score = float(_model.predict_proba(df_scaled)[0][1])
    else:
        score = float(_model.predict(df_scaled)[0])

    score = round(score, 4)
    if return_features:
        return score, features
    return score


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [--benchmark]", file=sys.stderr)
        sys.exit(1)

    img_path = sys.argv[1]

    if not Path(img_path).exists():
        print(f"Error: file not found: {img_path}", file=sys.stderr)
        sys.exit(1)

    # Cold-start timing
    t0 = time.perf_counter()
    score = predict(img_path)
    cold_ms = (time.perf_counter() - t0) * 1000

    print(score)

    if "--benchmark" in sys.argv:
        # Warm latency
        warm_times = []
        for _ in range(5):
            t1 = time.perf_counter()
            predict(img_path)
            warm_times.append((time.perf_counter() - t1) * 1000)

        print(
            f"\n[benchmark] mode=Handcrafted-only (no PyTorch/timm)\n"
            f"  cold (first call): {cold_ms:.0f} ms\n"
            f"  warm: mean={np.mean(warm_times):.1f} ms  "
            f"min={np.min(warm_times):.1f} ms  max={np.max(warm_times):.1f} ms",
            file=sys.stderr,
        )
