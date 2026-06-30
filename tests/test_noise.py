from cv.preprocessing import ImagePreprocessor
from cv.noise import NoiseFeatureExtractor
from tests.utils import get_test_image

pre = ImagePreprocessor()

data = pre.preprocess(get_test_image())

noise = NoiseFeatureExtractor()

result = noise.extract(data["enhanced"])

print(result)
