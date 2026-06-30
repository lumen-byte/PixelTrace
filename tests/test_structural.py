from cv.preprocessing import ImagePreprocessor
from cv.structural import StructuralFeatureExtractor
from tests.utils import get_test_image

pre = ImagePreprocessor()

data = pre.preprocess(get_test_image())

extractor = StructuralFeatureExtractor()

result = extractor.extract(data["enhanced"])

print(result)