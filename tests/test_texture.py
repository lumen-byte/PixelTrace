from cv.preprocessing import ImagePreprocessor
from cv.texture import TextureFeatureExtractor
from tests.utils import get_test_image

pre = ImagePreprocessor()

data = pre.preprocess(get_test_image())

texture = TextureFeatureExtractor()

result = texture.extract(data["enhanced"])

print(result)