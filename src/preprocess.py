import os
import configparser
import numpy as np
from glob import glob
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from logger import get_logger

logger = get_logger("preprocess")

config = configparser.ConfigParser()
config.read("config.ini")

DATA_PATH    = config["DATA"]["data_path"]
IMAGE_SIZE   = int(config["DATA"]["image_size"])
N_SAMPLES    = int(config["DATA"]["n_samples"])
TRAIN_SIZE   = float(config["DATA"]["train_size"])
HOG_ORIENT   = int(config["FEATURES"]["hog_orientations"])
HOG_PPC      = int(config["FEATURES"]["hog_pixels_per_cell"])
HOG_CPB      = int(config["FEATURES"]["hog_cells_per_block"])
SCALER_PATH  = config["PATHS"]["scaler_path"]
TEST_PATH    = config["PATHS"]["test_data_path"]
RANDOM_STATE = int(config["MODEL"]["random_state"])


def extract_hog_features(image_path):
    img = Image.open(image_path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_gray = rgb2gray(np.array(img))
    return hog(
        img_gray,
        orientations=HOG_ORIENT,
        pixels_per_cell=(HOG_PPC, HOG_PPC),
        cells_per_block=(HOG_CPB, HOG_CPB),
        block_norm="L2-Hys"
    )


def load_data():
    n_each = N_SAMPLES // 2
    cat_files = sorted(glob(os.path.join(DATA_PATH, "train", "cat.*.jpg")))[:n_each]
    dog_files = sorted(glob(os.path.join(DATA_PATH, "train", "dog.*.jpg")))[:n_each]

    all_files  = cat_files + dog_files
    all_labels = [0] * len(cat_files) + [1] * len(dog_files)

    logger.info(f"Загружаем {len(all_files)} изображений...")

    X = np.array([extract_hog_features(p) for p in all_files])
    y = np.array(all_labels)

    logger.info(f"HOG-матрица: {X.shape}")
    return X, y


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - TRAIN_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    os.makedirs("experiments", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump({"X_train": X_train, "y_train": y_train,
                 "X_test": X_test,  "y_test": y_test}, TEST_PATH)

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info("Данные сохранены")


if __name__ == "__main__":
    main()
