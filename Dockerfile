# Serves the trained wafer models (results/) through FastAPI. CPU-only PyTorch keeps the image small.
FROM python:3.11-slim

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY wafer_yield/ wafer_yield/
COPY results/cnn.pt results/gbt.joblib results/ensemble.json results/embeddings.npy \
     results/embedding_index.parquet results/lot_spc.parquet results/

RUN useradd --uid 10001 --no-create-home app
USER 10001
ENV WAFER_ROOT=/app OMP_NUM_THREADS=1 TORCH_THREADS=1 WEB_CONCURRENCY=1
EXPOSE 8000
CMD ["uvicorn", "wafer_yield.api:app", "--host", "0.0.0.0", "--port", "8000"]
