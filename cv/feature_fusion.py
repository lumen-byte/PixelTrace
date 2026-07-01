"""
PixelTrace - Feature Fusion Engine
----------------------------------
Combines features from all computer vision extractors into one
unified feature vector. FFT extractor now includes moiré features.
"""

from cv.edge import EdgeFeatureExtractor
from cv.texture import TextureFeatureExtractor
from cv.structural import StructuralFeatureExtractor
from cv.reflection import ReflectionFeatureExtractor
from cv.sharpness import SharpnessFeatureExtractor
from cv.geometry import GeometryFeatureExtractor
from cv.noise import NoiseFeatureExtractor
from cv.chromatic import ChromaticAberrationExtractor


class FeatureFusionEngine:
    """
    Runs every feature extractor and returns
    one merged numerical feature dictionary.
    """

    def __init__(self):

        # (extractor, input_key) pairs — optimized for maximum speed
        # FFT and Color extractors are removed to achieve sub-20ms warm local latency.
        self._pipeline = [
            (EdgeFeatureExtractor(),          "enhanced"),
            (TextureFeatureExtractor(),       "enhanced"),
            (StructuralFeatureExtractor(),    "enhanced"),
            (ReflectionFeatureExtractor(),    "resized"),
            (SharpnessFeatureExtractor(),     "enhanced"),
            (GeometryFeatureExtractor(),      "enhanced"),
            (NoiseFeatureExtractor(),         "enhanced"),
            (ChromaticAberrationExtractor(),  "resized"),
        ]

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

        return features