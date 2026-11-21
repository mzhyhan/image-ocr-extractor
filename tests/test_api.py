"""Tests for the FastAPI web demo and REST API."""

from __future__ import annotations

import io
import shutil

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_extractor import __version__
from ocr_extractor.ocr import OCRError
from ocr_extractor.web.app import app

from .conftest import SAMPLE_TEXT

client = TestClient(app)
needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None, reason="tesseract not installed"
)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_index_page() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Image <span>OCR</span> Extractor" in response.text


def test_health() -> None:
    assert client.get("/health").json()["status"] == "ok"


@needs_tesseract
def test_languages() -> None:
    assert "eng" in client.get("/languages").json()["languages"]


@needs_tesseract
def test_ocr_endpoint(text_image: Image.Image) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("sample.png", _png(text_image), "image/png")},
        data={"lang": "eng", "threshold": "adaptive"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == SAMPLE_TEXT
    assert body["filename"] == "sample.png"
    assert body["word_count"] == 3
    assert "threshold:adaptive" in body["preprocessing"]


@needs_tesseract
def test_ocr_without_preprocessing(text_image: Image.Image) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("sample.png", _png(text_image), "image/png")},
        data={"preprocess": "false"},
    )
    assert response.status_code == 200
    assert response.json()["preprocessing"] == []


def test_rejects_non_image() -> None:
    response = client.post("/ocr", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415


@needs_tesseract
def test_rejects_unknown_language(text_image: Image.Image) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("sample.png", _png(text_image), "image/png")},
        data={"lang": "xx_bad"},
    )
    assert response.status_code == 422


def test_rejects_oversized_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ocr_extractor.web.app.MAX_UPLOAD_BYTES", 10)
    response = client.post("/ocr", files={"file": ("big.png", b"x" * 100, "image/png")})
    assert response.status_code == 413


def test_languages_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing() -> list[str]:
        raise OCRError("Tesseract is not installed or not on PATH")

    monkeypatch.setattr("ocr_extractor.web.app.available_languages", _missing)
    assert client.get("/languages").status_code == 503


def test_version_is_exposed() -> None:
    assert client.get("/health").json()["version"] == __version__
