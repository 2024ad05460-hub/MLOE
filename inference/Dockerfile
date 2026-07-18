# LogiEdge inference container.
#
# Dependency and application layers are created before the model layer so
# a model-only OTA update reuses all earlier Docker cache layers.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONPATH=/app

ENV MODEL_PATH=/data/model.tflite
ENV STATS_PATH=/data/training_stats.npy
ENV REFERENCE_PATH=/data/reference_dist.json
ENV ALERT_DB=/data/alerts.db

ENV TRUCK_ID=TRK-01
ENV LOCAL_MQTT_HOST=127.0.0.1
ENV LOCAL_MQTT_PORT=1883
ENV UPLINK_MQTT_HOST=127.0.0.1
ENV UPLINK_MQTT_PORT=1884

WORKDIR /app

# ---------------------------------------------------------------------------
# Dependency layer
# ---------------------------------------------------------------------------
# Keep dependency installation before application and model copies.
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir \
    -r /app/requirements.txt

# ---------------------------------------------------------------------------
# Application layer
# ---------------------------------------------------------------------------
COPY preprocessing.py /app/preprocessing.py
COPY tflite_eval.py /app/tflite_eval.py
COPY psi.py /app/psi.py
COPY inference_service.py /app/inference_service.py

# ---------------------------------------------------------------------------
# Runtime data layer
# ---------------------------------------------------------------------------
RUN mkdir -p /data

COPY training_stats.npy /data/training_stats.npy
COPY reference_dist.json /data/reference_dist.json

# ---------------------------------------------------------------------------
# Model layer
# ---------------------------------------------------------------------------
# Keep model.tflite as the final changing file layer for efficient OTA updates.
COPY model.tflite /data/model.tflite

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=20s \
    --retries=3 \
    CMD python -c "import os, sys; required = [os.environ['MODEL_PATH'], os.environ['STATS_PATH'], os.environ['REFERENCE_PATH']]; sys.exit(0 if all(os.path.isfile(path) for path in required) else 1)"

CMD ["python", "/app/inference_service.py"]