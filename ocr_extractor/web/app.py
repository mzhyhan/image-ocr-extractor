"""FastAPI application exposing OCR over HTTP plus a small upload page.

Run locally with::

    uvicorn ocr_extractor.web.app:app --reload

then open http://localhost:8000 for the web UI or http://localhost:8000/docs for the
interactive OpenAPI documentation.

Endpoints:
    - ``GET /`` - drag-and-drop upload page.
    - ``POST /ocr`` - extract text from an uploaded image.
    - ``GET /languages`` - installed Tesseract languages.
    - ``GET /health`` - liveness probe.
"""

from __future__ import annotations

import io
import logging
from importlib import resources
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from ocr_extractor import __version__
from ocr_extractor.ocr import OCRError, available_languages, extract_text
from ocr_extractor.preprocessing import PreprocessConfig, ThresholdMethod

__all__ = ["MAX_UPLOAD_BYTES", "app"]

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
"""Largest accepted upload (10 MB)."""

app = FastAPI(
    title="Image OCR Extractor",
    description="Extract text from images with OpenCV preprocessing and Tesseract OCR.",
    version=__version__,
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """Serve the single-page upload UI.

    Returns:
        The HTML of ``static/index.html``.
    """
    return resources.files("ocr_extractor.web").joinpath("static/index.html").read_text("utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    """Report that the service is running.

    Returns:
        ``{"status": "ok", "version": "<package version>"}``.
    """
    return {"status": "ok", "version": __version__}


@app.get("/languages")
def languages() -> dict[str, list[str]]:
    """List the OCR languages available on the server.

    Returns:
        ``{"languages": [...]}`` with installed Tesseract language codes.

    Raises:
        HTTPException: ``503`` if Tesseract is not available.
    """
    try:
        return {"languages": available_languages()}
    except OCRError as exc:
        logger.error("Tesseract unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/ocr")
async def ocr(
    file: Annotated[UploadFile, File(description="Image to extract text from")],
    lang: Annotated[str, Form(description="Tesseract language(s), e.g. eng+ukr")] = "eng",
    preprocess: Annotated[bool, Form(description="Run the preprocessing pipeline")] = True,
    threshold: Annotated[ThresholdMethod, Form(description="Binarisation method")] = "otsu",
) -> dict[str, Any]:
    """Extract text from an uploaded image.

    Args:
        file: The uploaded image (PNG, JPEG, TIFF, BMP, WebP, ...).
        lang: Tesseract language code(s).
        preprocess: Whether to clean the image before OCR.
        threshold: Binarisation method used during preprocessing.

    Returns:
        The :meth:`~ocr_extractor.ocr.OCRResult.to_dict` payload plus ``filename``.

    Raises:
        HTTPException: ``413`` if the file exceeds 10 MB, ``415`` if it is not an image,
            ``422`` if OCR fails (e.g. unknown language).
    """
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 10 MB")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=415, detail="File is not a supported image") from exc

    try:
        result = extract_text(
            image,
            lang=lang,
            preprocess_image=preprocess,
            config=PreprocessConfig(threshold=threshold),
        )
    except OCRError as exc:
        logger.warning("OCR failed for %s: %s", file.filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"filename": file.filename, **result.to_dict()}
