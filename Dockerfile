FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir setuptools && pip install --no-cache-dir -r requirements.txt

COPY config.ini .
COPY src/ ./src/

EXPOSE 5000

CMD ["python", "src/api.py"]
