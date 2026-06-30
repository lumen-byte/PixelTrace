from cv.preprocessing import ImagePreprocessor
from cv.sharpness import SharpnessFeatureExtractor
from tests.utils import get_test_image

pre = ImagePreprocessor()

data = pre.preprocess(get_test_image())

sharpness = SharpnessFeatureExtractor()

result = sharpness.extract(data["enhanced"])

print(result)