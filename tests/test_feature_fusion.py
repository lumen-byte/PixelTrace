from cv.preprocessing import ImagePreprocessor
from cv.feature_fusion import FeatureFusionEngine
from tests.utils import get_test_image

pre = ImagePreprocessor()

data = pre.preprocess(get_test_image())

fusion = FeatureFusionEngine()

features = fusion.extract(data)

print("=" * 40)
print(f"Total Features : {len(features)}")
print("=" * 40)

for key, value in sorted(features.items()):
    print(f"{key:<30} {value}")