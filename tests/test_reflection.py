from cv.preprocessing import ImagePreprocessor
from cv.reflection import ReflectionFeatureExtractor
from tests.utils import get_test_image

pre = ImagePreprocessor()

data = pre.preprocess(get_test_image())

reflection = ReflectionFeatureExtractor()

result = reflection.extract(data["resized"])

print(result)