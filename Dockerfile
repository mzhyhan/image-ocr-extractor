FROM python:3.12-slim

LABEL org.opencontainers.image.title="image-ocr-extractor" \
      org.opencontainers.image.description="OCR web API: OpenCV preprocessing + Tesseract" \
      org.opencontainers.image.source="https://github.com/mzhyhan/image-ocr-extractor" \
      org.opencontainers.image.licenses="MIT"

# Tesseract OCR engine + English and Ukrainian language packs
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng tesseract-ocr-ukr \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY ocr_extractor ./ocr_extractor
RUN pip install ".[web]"

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "ocr_extractor.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
