from cv.preprocessing import ImagePreprocessor
from cv.color import ColorFeatureExtractor
from tests.utils import get_test_image

pre = ImagePreprocessor()

data = pre.preprocess(get_test_image())

color = ColorFeatureExtractor()

result = color.extract(data["resized"])

print(result)