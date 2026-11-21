"""Tests for the Tesseract wrapper and structured OCR results."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytesseract
import pytest
from PIL import Image

from ocr_extractor import extract_text
from ocr_extractor.ocr import OCRError, available_languages, load_image

from .conftest import SAMPLE_TEXT

pytestmark = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")


def test_extracts_text_from_path(text_image_path: Path) -> None:
    result = extract_text(text_image_path)
    assert result.text == SAMPLE_TEXT
    assert result.lang == "eng"


def test_without_preprocessing(text_image: Image.Image) -> None:
    result = extract_text(text_image, preprocess_image=False)
    assert result.text == SAMPLE_TEXT
    assert result.preprocessing == []


def test_noisy_image_is_cleaned_up(noisy_image: Image.Image) -> None:
    result = extract_text(noisy_image)
    assert "OCR" in result.text
    assert "denoise" in result.preprocessing


def test_word_level_output(text_image: Image.Image) -> None:
    result = extract_text(text_image)
    assert [w.text for w in result.words] == SAMPLE_TEXT.split()
    assert all(0 <= w.confidence <= 100 for w in result.words)
    assert all(w.bbox.width > 0 and w.bbox.height > 0 for w in result.words)
    assert result.mean_confidence > 70

    payload = result.to_dict()
    assert payload["word_count"] == 3
    assert set(payload["words"][0]) == {"text", "confidence", "bbox", "line"}


def test_unknown_language_raises(text_image: Image.Image) -> None:
    with pytest.raises(OCRError):
        extract_text(text_image, lang="xx_not_a_language")


def test_available_languages_includes_english() -> None:
    assert "eng" in available_languages()


def test_rgba_images_are_normalised(text_image: Image.Image) -> None:
    assert load_image(text_image.convert("RGBA")).mode == "RGB"


def test_text_only_mode_skips_words(text_image: Image.Image) -> None:
    result = extract_text(text_image, with_words=False)
    assert result.text == SAMPLE_TEXT
    assert result.words == []
    assert result.mean_confidence == 0.0


def test_missing_tesseract_is_reported(
    text_image: Image.Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _missing(*_args: object, **_kwargs: object) -> None:
        raise pytesseract.TesseractNotFoundError()

    monkeypatch.setattr(pytesseract, "image_to_string", _missing)
    monkeypatch.setattr(pytesseract, "get_languages", _missing)
    with pytest.raises(OCRError, match="not installed"):
        extract_text(text_image)
    with pytest.raises(OCRError, match="not installed"):
        available_languages()
