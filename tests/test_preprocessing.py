"""Tests for the OpenCV preprocessing pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ocr_extractor.preprocessing import (
    PreprocessConfig,
    estimate_skew,
    pil_to_cv,
    preprocess,
    rotate,
    to_grayscale,
)


def test_preprocess_returns_binary_grayscale(text_image: Image.Image) -> None:
    result = preprocess(text_image)
    assert result.image.mode == "L"
    assert set(np.unique(np.array(result.image))) <= {0, 255}
    assert result.steps[0] == "grayscale"
    assert "threshold:otsu" in result.steps


def test_small_images_are_upscaled(text_image: Image.Image) -> None:
    result = preprocess(text_image, PreprocessConfig(min_height=400))
    assert result.image.height >= 400
    assert any(step.startswith("resize") for step in result.steps)


def test_resize_can_be_disabled(text_image: Image.Image) -> None:
    result = preprocess(text_image, PreprocessConfig(min_height=0, deskew=False))
    assert result.image.size == text_image.size


@pytest.mark.parametrize("method", ["otsu", "adaptive", "none"])
def test_threshold_methods(text_image: Image.Image, method: str) -> None:
    config = PreprocessConfig(threshold=method, denoise=False)  # type: ignore[arg-type]
    result = preprocess(text_image, config)
    assert result.image.mode == "L"


def test_deskew_detects_rotation(text_image: Image.Image) -> None:
    gray = to_grayscale(pil_to_cv(text_image.resize((text_image.width * 2, text_image.height * 2))))
    rotated = rotate(gray, 5.0)
    assert estimate_skew(rotated) == pytest.approx(5.0, abs=1.5)


def test_blank_image_has_no_skew() -> None:
    blank = np.full((100, 100), 255, dtype=np.uint8)
    assert estimate_skew(blank) == 0.0


def test_colour_pipeline_without_grayscale(text_image: Image.Image) -> None:
    config = PreprocessConfig(grayscale=False, threshold="none", deskew=False, min_height=0)
    result = preprocess(text_image, config)
    assert result.steps == ["denoise"]
    # Thresholding was disabled, so the colour image keeps its three channels.
    assert result.image.mode == "RGB"


def test_large_skew_is_ignored(text_image: Image.Image) -> None:
    result = preprocess(text_image, PreprocessConfig(max_skew_degrees=0.0))
    assert result.skew_angle == 0.0
    assert not any(step.startswith("deskew") for step in result.steps)
