"""Shared fixtures: sample images are generated on the fly with Pillow (no binaries in git)."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

SAMPLE_TEXT = "Hello OCR World"
"""Text rendered into every generated test image."""


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return DejaVu Sans if available (installed on most Linux systems), else Pillow's default."""
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def render_text(text: str, size: int = 48, padding: int = 40) -> Image.Image:
    """Render black ``text`` on a white canvas sized to fit it.

    Args:
        text: The text to draw.
        size: Font size in pixels.
        padding: White margin around the text in pixels.

    Returns:
        An RGB image containing the rendered text.
    """
    font = _font(size)
    left, top, right, bottom = (int(v) for v in font.getbbox(text))
    image = Image.new("RGB", (right - left + 2 * padding, bottom - top + 2 * padding), "white")
    ImageDraw.Draw(image).text((padding - left, padding - top), text, fill="black", font=font)
    return image


@pytest.fixture
def text_image() -> Image.Image:
    """A clean, horizontal image of :data:`SAMPLE_TEXT`."""
    return render_text(SAMPLE_TEXT)


@pytest.fixture
def text_image_path(tmp_path: Path, text_image: Image.Image) -> Path:
    """:func:`text_image` saved as a PNG file in a temporary directory."""
    path = tmp_path / "sample.png"
    text_image.save(path)
    return path


@pytest.fixture
def noisy_image(text_image: Image.Image) -> Image.Image:
    """:func:`text_image` in grayscale with heavy Gaussian noise (seeded, deterministic)."""
    import numpy as np

    rng = np.random.default_rng(42)
    array = np.array(text_image.convert("L")).astype(np.int16)
    array += rng.normal(0, 25, array.shape).astype(np.int16)
    return Image.fromarray(np.clip(array, 0, 255).astype("uint8"))
