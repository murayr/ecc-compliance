# API image: FastAPI + retrieval index + bge-m3 embedding model.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.hf

# CPU-only PyTorch: the CUDA build is ~2 GB larger and unused here.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download bge-m3 once: the PyTorch weights only, not the 2.2 GB ONNX copy. Then go offline, otherwise
# transformers also fetches a converted model.safetensors (another 2.2 GB) in the background.
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', \
    ignore_patterns=['onnx/*', '*.onnx', 'imgs/*', 'colbert_linear.pt', 'sparse_linear.pt'])"
ENV HF_HUB_OFFLINE=1

COPY ingest.py rag.py api.py ./
COPY data/ data/
# Build the index at image build time, so container startup is fast.
RUN python ingest.py

EXPOSE 8000
# ANTHROPIC_API_KEY and CLAUDE_MODEL come from the environment at run time, never baked in.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
