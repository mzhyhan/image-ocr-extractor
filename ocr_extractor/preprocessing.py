"""Image preprocessing pipeline that prepares images for Tesseract OCR.

Tesseract performs best on clean, level, black-on-white text whose glyphs are
roughly 20-30 px tall. Real-world photos and scans rarely look like that, so
:func:`preprocess` applies a configurable sequence of OpenCV operations:

1. **Grayscale** - discard colour; OCR only needs luminance.
2. **Resize** - upscale small images so glyphs have enough pixels.
3. **Denoise** - non-local means denoising removes sensor and JPEG noise.
4. **Deskew** - estimate the text rotation and rotate the page level.
5. **Threshold** - binarise with Otsu (global) or adaptive (local) thresholding.

Example:
    >>> from PIL import Image
    >>> from ocr_extractor.preprocessing import PreprocessConfig, preprocess
    >>> image = Image.new("RGB", (200, 50), "white")
    >>> result = preprocess(image, PreprocessConfig(min_height=100))
    >>> result.steps[:2]
    ['grayscale', 'resize x2.00']
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

__all__ = [
    "PreprocessConfig",
    "PreprocessResult",
    "ThresholdMethod",
    "cv_to_pil",
    "denoise",
    "estimate_skew",
    "pil_to_cv",
    "preprocess",
    "resize_to_min_height",
    "rotate",
    "threshold",
    "to_grayscale",
]

logger = logging.getLogger(__name__)

ThresholdMethod = Literal["otsu", "adaptive", "none"]
"""Supported binarisation strategies."""

ImageArray = NDArray[np.uint8]
"""An OpenCV image: ``(H, W)`` grayscale or ``(H, W, 3)`` BGR, dtype ``uint8``."""

_MIN_SKEW_DEGREES = 0.1
_MIN_TEXT_PIXELS = 50


def _as_u8(array: Any) -> ImageArray:
    """Narrow an OpenCV return value to a ``uint8`` array (no copy if already ``uint8``)."""
    return np.asarray(array, dtype=np.uint8)


@dataclass(frozen=True)
class PreprocessConfig:
    """Settings that control which preprocessing steps run and how.

    Attributes:
        grayscale: Convert the image to a single luminance channel.
        min_height: Upscale images shorter than this many pixels. ``0`` disables resizing.
        denoise: Apply non-local means denoising.
        threshold: Binarisation method: ``"otsu"``, ``"adaptive"`` or ``"none"``.
        deskew: Detect and correct page rotation.
        max_skew_degrees: Ignore detected skew larger than this (likely a false estimate).
    """

    grayscale: bool = True
    min_height: int = 600
    denoise: bool = True
    threshold: ThresholdMethod = "otsu"
    deskew: bool = True
    max_skew_degrees: float = 45.0


@dataclass
class PreprocessResult:
    """Output of :func:`preprocess`.

    Attributes:
        image: The processed image, ready to be passed to Tesseract.
        steps: Human-readable log of the steps that were applied, in order.
        skew_angle: Detected and corrected skew in degrees (``0.0`` if none was applied).
    """

    image: Image.Image
    steps: list[str] = field(default_factory=list)
    skew_angle: float = 0.0


def pil_to_cv(image: Image.Image) -> ImageArray:
    """Convert a Pillow image to an OpenCV array.

    Args:
        image: Any Pillow image. Grayscale and bilevel images stay single-channel;
            everything else is converted to 3-channel BGR.

    Returns:
        A ``uint8`` NumPy array in OpenCV channel order.
    """
    if image.mode in ("L", "1"):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return _as_u8(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def cv_to_pil(array: ImageArray) -> Image.Image:
    """Convert an OpenCV array back to a Pillow image.

    Args:
        array: A grayscale ``(H, W)`` or BGR ``(H, W, 3)`` array.

    Returns:
        A Pillow image in ``"L"`` or ``"RGB"`` mode.
    """
    if array.ndim == 2:
        return Image.fromarray(array)
    return Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))


def to_grayscale(array: ImageArray) -> ImageArray:
    """Return a single-channel version of ``array`` (no-op if already grayscale).

    Args:
        array: A grayscale or BGR image.

    Returns:
        A grayscale image.
    """
    if array.ndim == 2:
        return array
    return _as_u8(cv2.cvtColor(array, cv2.COLOR_BGR2GRAY))


def resize_to_min_height(array: ImageArray, min_height: int) -> ImageArray:
    """Upscale ``array`` with bicubic interpolation so it is at least ``min_height`` tall.

    Images that are already tall enough are returned unchanged (same object), and
    images are never downscaled.

    Args:
        array: The image to resize.
        min_height: Target minimum height in pixels. ``0`` or less disables resizing.

    Returns:
        The resized image, or ``array`` itself if no resize was needed.
    """
    height = array.shape[0]
    if min_height <= 0 or height >= min_height:
        return array
    scale = min_height / height
    return _as_u8(cv2.resize(array, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))


def denoise(array: ImageArray) -> ImageArray:
    """Remove noise while preserving edges using OpenCV non-local means denoising.

    Args:
        array: A grayscale or BGR image.

    Returns:
        The denoised image with the same shape as the input.
    """
    if array.ndim == 2:
        return _as_u8(cv2.fastNlMeansDenoising(array, None, 10, 7, 21))
    return _as_u8(cv2.fastNlMeansDenoisingColored(array, None, 10, 10, 7, 21))


def threshold(array: ImageArray, method: ThresholdMethod) -> ImageArray:
    """Binarise an image to pure black and white.

    Args:
        array: A grayscale or BGR image.
        method: ``"otsu"`` picks one global threshold automatically and works well for
            evenly lit scans. ``"adaptive"`` computes a local Gaussian threshold, which
            copes with shadows and uneven lighting. ``"none"`` only converts to grayscale.

    Returns:
        A single-channel image whose pixels are 0 or 255 (or grayscale for ``"none"``).
    """
    gray = to_grayscale(array)
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return _as_u8(binary)
    if method == "adaptive":
        return _as_u8(
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
            )
        )
    return gray


def estimate_skew(gray: ImageArray) -> float:
    """Estimate the rotation of dark text on a light background.

    The text pixels are isolated with Otsu thresholding and fitted with a
    minimum-area rectangle; the rectangle's angle is the page skew.

    Args:
        gray: A grayscale image with dark text on a light background.

    Returns:
        The skew in degrees in the range ``[-45, 45]``. Positive values mean the text
        is rotated counter-clockwise. Returns ``0.0`` when there is too little text.
    """
    inverted = cv2.bitwise_not(gray)
    _, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(mask > 0))
    if len(coords) < _MIN_TEXT_PIXELS:
        return 0.0
    # minAreaRect expects (x, y) points, np.where yields (row, col).
    angle = float(cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1])
    # OpenCV >= 4.5 reports angles in [0, 90); map them to [-45, 45].
    if angle > 45:
        angle -= 90
    return -angle


def rotate(array: ImageArray, angle: float) -> ImageArray:
    """Rotate an image about its centre, filling exposed corners with white.

    Args:
        array: A grayscale or BGR image.
        angle: Rotation in degrees; positive values rotate counter-clockwise.

    Returns:
        The rotated image with the same dimensions as the input.
    """
    height, width = array.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    border: int | tuple[int, int, int] = 255 if array.ndim == 2 else (255, 255, 255)
    rotated = cv2.warpAffine(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )
    return _as_u8(rotated)


def preprocess(image: Image.Image, config: PreprocessConfig | None = None) -> PreprocessResult:
    """Run the full preprocessing pipeline on an image.

    Args:
        image: The input image.
        config: Which steps to run. Defaults to :class:`PreprocessConfig` defaults
            (all steps enabled, Otsu thresholding).

    Returns:
        A :class:`PreprocessResult` with the cleaned image and the list of applied steps.

    Example:
        >>> from PIL import Image
        >>> out = preprocess(Image.new("L", (800, 800), 255))
        >>> out.steps
        ['grayscale', 'denoise', 'threshold:otsu']
    """
    config = config or PreprocessConfig()
    array = pil_to_cv(image)
    steps: list[str] = []
    skew = 0.0

    if config.grayscale:
        array = to_grayscale(array)
        steps.append("grayscale")

    resized = resize_to_min_height(array, config.min_height)
    if resized is not array:
        steps.append(f"resize x{resized.shape[0] / array.shape[0]:.2f}")
        array = resized

    if config.denoise:
        array = denoise(array)
        steps.append("denoise")

    if config.deskew:
        skew = estimate_skew(to_grayscale(array))
        if _MIN_SKEW_DEGREES < abs(skew) <= config.max_skew_degrees:
            array = rotate(array, -skew)
            steps.append(f"deskew {skew:+.1f}deg")
        else:
            skew = 0.0

    if config.threshold != "none":
        array = threshold(array, config.threshold)
        steps.append(f"threshold:{config.threshold}")

    logger.debug("Preprocessing applied: %s", ", ".join(steps) or "none")
    return PreprocessResult(image=cv_to_pil(array), steps=steps, skew_angle=skew)
