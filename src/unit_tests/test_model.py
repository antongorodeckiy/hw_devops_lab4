import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import joblib
import configparser
from sklearn.metrics import accuracy_score

config = configparser.ConfigParser()
config.read("config.ini")

MODEL_PATH = config["PATHS"]["model_path"]
TEST_PATH  = config["PATHS"]["test_data_path"]

model = joblib.load(MODEL_PATH)
data  = joblib.load(TEST_PATH)
X_test, y_test = data["X_test"], data["y_test"]


def test_model_loads():
    assert model is not None


def test_prediction_shape():
    pred = model.predict(X_test)
    assert len(pred) == len(y_test)


def test_prediction_labels():
    pred = model.predict(X_test)
    assert set(pred).issubset({0, 1})


def test_accuracy():
    acc = accuracy_score(y_test, model.predict(X_test))
    assert acc > 0.6


def test_predict_proba():
    proba = model.predict_proba(X_test[:5])
    assert proba.shape == (5, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
