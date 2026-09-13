FROM python:3.11-slim

WORKDIR /app
COPY deployment/requirements.txt /app/deployment/requirements.txt
RUN pip install --no-cache-dir -r deployment/requirements.txt

COPY deployment /app/deployment
COPY final_model /app/final_model
COPY datasets /app/datasets
COPY experiments/implementations /app/experiments/implementations
COPY experiments/__init__.py /app/experiments/__init__.py
COPY inference_options.py /app/inference_options.py
COPY models/final_study /app/models/final_study
COPY web /app/web

EXPOSE 8000
CMD ["uvicorn", "deployment.inference_server:app", "--host", "0.0.0.0", "--port", "8000"]
