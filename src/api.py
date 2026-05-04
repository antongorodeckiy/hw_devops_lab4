import io
import os
import configparser
import numpy as np
import joblib
import psycopg2
import hvac
from flask import Flask, request, jsonify, render_template
from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2gray
from logger import get_logger

app = Flask(__name__)
logger = get_logger("api")


def get_vault_secrets():
    client = hvac.Client(
        url=os.environ["VAULT_ADDR"],
        token=os.environ["VAULT_TOKEN"]
    )
    secret = client.secrets.kv.v2.read_secret_version(path="db", mount_point="secret")
    return secret["data"]["data"]

config = configparser.ConfigParser()
config.read("config.ini")

IMAGE_SIZE  = int(config["DATA"]["image_size"])
HOG_ORIENT  = int(config["FEATURES"]["hog_orientations"])
HOG_PPC     = int(config["FEATURES"]["hog_pixels_per_cell"])
HOG_CPB     = int(config["FEATURES"]["hog_cells_per_block"])
MODEL_PATH  = config["PATHS"]["model_path"]
SCALER_PATH = config["PATHS"]["scaler_path"]
HOST        = config["API"]["host"]
PORT        = int(config["API"]["port"])

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def get_db_connection():
    secrets = get_vault_secrets()
    return psycopg2.connect(
        host=secrets["POSTGRES_HOST"],
        port=secrets["POSTGRES_PORT"],
        dbname=secrets["POSTGRES_DB"],
        user=secrets["POSTGRES_USER"],
        password=secrets["POSTGRES_PASSWORD"]
    )


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            filename TEXT,
            label TEXT,
            confidence FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


def save_prediction(filename, label, confidence):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO predictions (filename, label, confidence) VALUES (%s, %s, %s)",
            (filename, label, confidence)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"DB error: {e}")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    try:
        if file.filename == "":
            raise ValueError("Empty file")
        img = Image.open(io.BytesIO(file.read())).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    except Exception as e:
        return jsonify({"error": f"Invalid image file: {e}"}), 400

    img_gray = rgb2gray(np.array(img))
    features = hog(
        img_gray,
        orientations=HOG_ORIENT,
        pixels_per_cell=(HOG_PPC, HOG_PPC),
        cells_per_block=(HOG_CPB, HOG_CPB),
        block_norm="L2-Hys"
    )

    features_scaled = scaler.transform([features])
    pred  = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]

    label = "dog" if pred == 1 else "cat"
    confidence = round(float(max(proba)), 3)
    logger.info(f"Предсказание: {label}, confidence: {confidence}")

    save_prediction(file.filename, label, confidence)

    return jsonify({
        "label": label,
        "confidence": confidence
    })


@app.route("/predictions", methods=["GET"])
def get_predictions():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id, filename, label, confidence, created_at FROM predictions ORDER BY created_at DESC LIMIT 100")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return jsonify([
            {"id": r[0], "filename": r[1], "label": r[2], "confidence": r[3], "created_at": str(r[4])}
            for r in rows
        ])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    init_db()
    app.run(host=HOST, port=PORT)
