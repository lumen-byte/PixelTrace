from cv.preprocessing import ImagePreprocessor
from cv.geometry import GeometryFeatureExtractor
from tests.utils import get_test_image

pre = ImagePreprocessor()

data = pre.preprocess(get_test_image())

extractor = GeometryFeatureExtractor()

result = extractor.extract(data["enhanced"])

print(result)