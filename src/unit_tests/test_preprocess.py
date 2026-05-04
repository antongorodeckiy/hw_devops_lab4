import sys
import os
import io
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from PIL import Image
from preprocess import extract_hog_features


def make_test_image():
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img.save(tmp.name)
    return tmp.name


def test_hog_output_type():
    path = make_test_image()
    features = extract_hog_features(path)
    os.unlink(path)
    assert isinstance(features, np.ndarray)


def test_hog_output_shape():
    path = make_test_image()
    features = extract_hog_features(path)
    os.unlink(path)
    assert features.ndim == 1
    assert features.shape[0] > 0


def test_hog_no_nan():
    path = make_test_image()
    features = extract_hog_features(path)
    os.unlink(path)
    assert not np.isnan(features).any()
