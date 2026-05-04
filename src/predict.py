import configparser
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from logger import get_logger

logger = get_logger("predict")

config = configparser.ConfigParser()
config.read("config.ini")

MODEL_PATH  = config["PATHS"]["model_path"]
TEST_PATH   = config["PATHS"]["test_data_path"]


def smoke_test(model, X_test, y_test):
    acc = accuracy_score(y_test, model.predict(X_test))
    logger.info(f"Smoke test accuracy: {acc:.4f}")
    assert acc > 0.6, f"Accuracy слишком низкая: {acc}"
    return acc


def functional_test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Cat", "Dog"])
    logger.info(f"\n{report}")
    return report


def main():
    model = joblib.load(MODEL_PATH)
    data  = joblib.load(TEST_PATH)
    X_test, y_test = data["X_test"], data["y_test"]

    smoke_test(model, X_test, y_test)
    functional_test(model, X_test, y_test)


if __name__ == "__main__":
    main()
