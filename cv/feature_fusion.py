"""
PixelTrace - Feature Fusion Engine
----------------------------------
Combines features from all computer vision extractors into one
unified feature vector. FFT extractor now includes moiré features.
"""

from cv.edge import EdgeFeatureExtractor
from cv.fft import FFTFeatureExtractor
from cv.texture import TextureFeatureExtractor
from cv.structural import StructuralFeatureExtractor
from cv.reflection import ReflectionFeatureExtractor
from cv.sharpness import SharpnessFeatureExtractor
from cv.color import ColorFeatureExtractor
from cv.geometry import GeometryFeatureExtractor
from cv.noise import NoiseFeatureExtractor
from cv.chromatic import ChromaticAberrationExtractor


class FeatureFusionEngine:
    """
    Runs every feature extractor and returns
    one merged numerical feature dictionary.
    """

    def __init__(self):

        # (extractor, input_key) pairs
        self._pipeline = [
            (EdgeFeatureExtractor(),          "enhanced"),
            (FFTFeatureExtractor(),           "gray"),      # FFT now includes moiré
            (TextureFeatureExtractor(),       "enhanced"),
            (StructuralFeatureExtractor(),    "enhanced"),
            (ReflectionFeatureExtractor(),    "resized"),
            (SharpnessFeatureExtractor(),     "enhanced"),
            (ColorFeatureExtractor(),         "resized"),
            (GeometryFeatureExtractor(),      "enhanced"),
            (NoiseFeatureExtractor(),         "enhanced"),
            (ChromaticAberrationExtractor(),  "resized"),
        ]

        # Load CNN feature extractor and SVD projection model
        import os
        if os.environ.get("PT_NO_CNN") == "1":
            self.cnn_extractor = None
            self.svd_model = None
        else:
            try:
                import joblib
                from ml.cnn_embedding import CNNEmbeddingExtractor
                self.svd_model = joblib.load("ml/models/cnn_svd.pkl")
                self.cnn_extractor = CNNEmbeddingExtractor()
            except Exception as e:
                print(f"[WARNING] Failed to load CNN embedding extractor or SVD: {e}")
                self.cnn_extractor = None
                self.svd_model = None

    def extract(self, preprocessed: dict) -> dict:

        features = {}

        for extractor, input_key in self._pipeline:
            try:
                result = extractor.extract(preprocessed[input_key])

                # Keep only numerical values
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        features[key] = value

            except Exception as e:
                print(
                    f"[WARNING] {extractor.__class__.__name__} failed: {e}"
                )

        # Extract CNN features and append them
        if self.cnn_extractor is not None and self.svd_model is not None:
            try:
                raw_emb = self.cnn_extractor.extract(preprocessed["original"])
                reduced_emb = self.svd_model.transform(raw_emb.reshape(1, -1)).squeeze(0)
                for i in range(32):
                    features[f"cnn_{i:03d}"] = float(reduced_emb[i])
            except Exception as e:
                print(f"[WARNING] CNN embedding extraction failed: {e}")

        return features