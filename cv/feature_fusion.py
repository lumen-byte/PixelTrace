"""
PixelTrace - Feature Fusion Engine
----------------------------------
Combines features from all computer vision extractors and the CNN
embedding model into one unified feature vector.
"""

import joblib
from cv.edge import EdgeFeatureExtractor
from cv.fft import FFTFeatureExtractor
from cv.texture import TextureFeatureExtractor
from cv.structural import StructuralFeatureExtractor
from cv.reflection import ReflectionFeatureExtractor
from cv.sharpness import SharpnessFeatureExtractor
from cv.color import ColorFeatureExtractor
from cv.geometry import GeometryFeatureExtractor
from cv.noise import NoiseFeatureExtractor
from cv.moire import MoireDetector
from cv.chromatic import ChromaticAberrationExtractor



class FeatureFusionEngine:
    """
    Runs every feature extractor and returns
    one merged numerical feature dictionary.
    """

    def __init__(self):

        self.extractors = [
            EdgeFeatureExtractor(),
            FFTFeatureExtractor(),
            TextureFeatureExtractor(),
            StructuralFeatureExtractor(),
            ReflectionFeatureExtractor(),
            SharpnessFeatureExtractor(),
            ColorFeatureExtractor(),
            GeometryFeatureExtractor(),
            NoiseFeatureExtractor(),
            MoireDetector(),
            ChromaticAberrationExtractor(),
        ]

        # Load CNN feature extractor and SVD projection model
        import os
        if os.environ.get("PT_NO_CNN") == "1":
            self.cnn_extractor = None
            self.svd_model = None
        else:
            try:
                from ml.cnn_embedding import CNNEmbeddingExtractor
                self.svd_model = joblib.load("ml/models/cnn_svd.pkl")
                self.cnn_extractor = CNNEmbeddingExtractor()
            except Exception as e:
                print(f"[WARNING] Failed to load CNN embedding extractor or SVD: {e}")
                self.cnn_extractor = None
                self.svd_model = None

    def extract(self, preprocessed: dict) -> dict:

        features = {}

        for extractor in self.extractors:

            try:

                # Determine proper input automatically
                if extractor.__class__.__name__ in [
                    "ReflectionFeatureExtractor",
                    "ColorFeatureExtractor",
                    "ChromaticAberrationExtractor",
                ]:
                    result = extractor.extract(
                        preprocessed["resized"]
                    )

                elif extractor.__class__.__name__ == "MoireDetector":
                    result = extractor.extract(
                        preprocessed["gray"]
                    )

                else:
                    result = extractor.extract(
                        preprocessed["enhanced"]
                    )

                # Keep only numerical values
                for key, value in result.items():

                    if isinstance(value, (int, float)):
                        features[key] = value

            except Exception as e:

                print(
                    f"[WARNING] {extractor.__class__.__name__} failed: {e}"
                )

        # Extract CNN features and append them in the requested order (cnn_000 to cnn_031)
        if self.cnn_extractor is not None and self.svd_model is not None:
            try:
                raw_emb = self.cnn_extractor.extract(preprocessed["original"])
                reduced_emb = self.svd_model.transform(raw_emb.reshape(1, -1)).squeeze(0)
                for i in range(32):
                    features[f"cnn_{i:03d}"] = float(reduced_emb[i])
            except Exception as e:
                print(f"[WARNING] CNN embedding extraction failed: {e}")

        return features