import sys
import os
import io
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from PIL import Image
from api import app


def make_test_image_bytes():
    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_predict_returns_200(client):
    response = client.post("/predict", data={"file": (make_test_image_bytes(), "test.jpg")})
    assert response.status_code == 200


def test_predict_response_structure(client):
    response = client.post("/predict", data={"file": (make_test_image_bytes(), "test.jpg")})
    data = response.get_json()
    assert "label" in data
    assert "confidence" in data


def test_predict_label_valid(client):
    response = client.post("/predict", data={"file": (make_test_image_bytes(), "test.jpg")})
    label = response.get_json()["label"]
    assert label in {"cat", "dog"}


def test_predict_no_file(client):
    response = client.post("/predict")
    assert response.status_code == 400
