FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-mal \
        tesseract-ocr-eng \
        libglib2.0-0 \
        libgl1 \
        libsm6 \
        libxext6 \
        libxrender1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY docs/ docs/
COPY src/ src/
COPY scripts/ scripts/

RUN pip install --no-cache-dir uv && uv pip install --system "."

ENV PYTHONUNBUFFERED=1
