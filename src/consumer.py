import os
import json
import time
import psycopg2
import hvac
from kafka import KafkaConsumer
from logger import get_logger

logger = get_logger("consumer")

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.environ.get("KAFKA_TOPIC", "predictions")


def get_vault_secrets():
    client = hvac.Client(
        url=os.environ["VAULT_ADDR"],
        token=os.environ["VAULT_TOKEN"]
    )
    secret = client.secrets.kv.v2.read_secret_version(path="db", mount_point="secret")
    return secret["data"]["data"]


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
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO predictions (filename, label, confidence) VALUES (%s, %s, %s)",
        (filename, label, confidence)
    )
    conn.commit()
    cur.close()
    conn.close()


def wait_for_kafka():
    while True:
        try:
            consumer = KafkaConsumer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                request_timeout_ms=5000
            )
            consumer.close()
            logger.info("Kafka is ready")
            return
        except Exception:
            logger.info("Waiting for Kafka...")
            time.sleep(3)


if __name__ == "__main__":
    wait_for_kafka()
    init_db()
    logger.info(f"Starting consumer on topic '{KAFKA_TOPIC}'")

    consumer = KafkaConsumer(
        KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="predictions-group"
    )

    for message in consumer:
        data = message.value
        try:
            save_prediction(data["filename"], data["label"], data["confidence"])
            logger.info(f"Saved: {data['label']} ({data['confidence']})")
        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
