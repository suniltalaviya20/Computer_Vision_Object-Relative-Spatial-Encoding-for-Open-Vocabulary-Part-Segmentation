FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY deployment/requirements-api.txt /app/deployment/requirements-api.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/deployment/requirements-api.txt

COPY datasets /app/datasets
COPY deployment /app/deployment
COPY final_training /app/final_training
COPY models/final_study /app/models/final_study
COPY web /app/web
COPY inference_server.py /app/inference_server.py

EXPOSE 8000

# One worker avoids loading a separate copy of the large vision encoders per process.
CMD ["sh", "-c", "uvicorn inference_server:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
