from cv.preprocessing import ImagePreprocessor
from cv.edge import EdgeFeatureExtractor
from tests.utils import get_test_image

pre = ImagePreprocessor()

data = pre.preprocess(get_test_image())

edge = EdgeFeatureExtractor()

result = edge.extract(data["enhanced"])

print(result["edge_density"])
print(result["edge_pixels"])