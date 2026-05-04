import configparser
import joblib
from sklearn.svm import SVC
from logger import get_logger

logger = get_logger("train")

config = configparser.ConfigParser()
config.read("config.ini")

KERNEL       = config["MODEL"]["kernel"]
C            = float(config["MODEL"]["C"])
GAMMA        = config["MODEL"]["gamma"]
RANDOM_STATE = int(config["MODEL"]["random_state"])
MODEL_PATH   = config["PATHS"]["model_path"]
TEST_PATH    = config["PATHS"]["test_data_path"]


def main():
    data = joblib.load(TEST_PATH)
    X_train, y_train = data["X_train"], data["y_train"]

    logger.info(f"Обучаем SVM (kernel={KERNEL}, C={C}, gamma={GAMMA})...")
    model = SVC(kernel=KERNEL, C=C, gamma=GAMMA,
                random_state=RANDOM_STATE, probability=True)
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    logger.info(f"Модель сохранена: {MODEL_PATH}")


if __name__ == "__main__":
    main()
