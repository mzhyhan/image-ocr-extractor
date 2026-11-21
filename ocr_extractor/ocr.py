"""Tesseract OCR wrapper returning plain text or structured, word-level results.

The main entry point is :func:`extract_text`, which optionally runs the
:mod:`ocr_extractor.preprocessing` pipeline before calling Tesseract and returns an
:class:`OCRResult` containing the text plus per-word confidence and bounding boxes.

Example:
    ::

        from ocr_extractor import extract_text

        result = extract_text("receipt.jpg", lang="eng")
        print(result.text)              # "COFFEE HOUSE LONDON ..."
        print(result.mean_confidence)   # 94.52
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image, ImageOps

from ocr_extractor.preprocessing import PreprocessConfig, preprocess

__all__ = [
    "BoundingBox",
    "OCRError",
    "OCRResult",
    "Word",
    "available_languages",
    "extract_text",
    "load_image",
]

logger = logging.getLogger(__name__)

ImageSource = str | Path | Image.Image
"""Anything :func:`load_image` accepts: a file path or an already loaded Pillow image."""


class OCRError(RuntimeError):
    """Raised when OCR cannot be performed.

    Typical causes are a missing Tesseract binary or an unknown language pack.
    """


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box around a word, in pixels of the image passed to Tesseract.

    Attributes:
        left: X coordinate of the left edge.
        top: Y coordinate of the top edge.
        width: Box width.
        height: Box height.
    """

    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class Word:
    """A single recognised word.

    Attributes:
        text: The recognised characters.
        confidence: Tesseract confidence in the range 0-100.
        bbox: Location of the word.
        line: Zero-based index of the text line the word belongs to.
    """

    text: str
    confidence: float
    bbox: BoundingBox
    line: int


@dataclass
class OCRResult:
    """The outcome of an OCR run.

    Attributes:
        text: Extracted text with line breaks preserved.
        lang: Tesseract language code(s) used.
        words: Word-level details (empty when ``with_words=False``).
        preprocessing: Preprocessing steps that were applied, in order.
    """

    text: str
    lang: str
    words: list[Word] = field(default_factory=list)
    preprocessing: list[str] = field(default_factory=list)

    @property
    def mean_confidence(self) -> float:
        """Average word confidence (0-100), rounded to two decimals; ``0.0`` if no words."""
        if not self.words:
            return 0.0
        return round(sum(w.confidence for w in self.words) / len(self.words), 2)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the result to JSON-compatible primitives.

        Returns:
            A dictionary with ``text``, ``lang``, ``mean_confidence``, ``word_count``,
            ``preprocessing`` and ``words`` keys.
        """
        return {
            "text": self.text,
            "lang": self.lang,
            "mean_confidence": self.mean_confidence,
            "word_count": len(self.words),
            "preprocessing": self.preprocessing,
            "words": [asdict(w) for w in self.words],
        }


def available_languages() -> list[str]:
    """List the Tesseract language packs installed on this machine.

    Returns:
        Sorted language codes, e.g. ``["eng", "osd", "ukr"]``.

    Raises:
        OCRError: If Tesseract is not installed or not on ``PATH``.
    """
    try:
        return sorted(pytesseract.get_languages(config=""))
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRError("Tesseract is not installed or not on PATH") from exc


def load_image(source: ImageSource) -> Image.Image:
    """Load an image and normalise its orientation and colour mode.

    EXIF orientation (common in phone photos) is applied, and unusual modes such as
    ``RGBA``, ``P`` or ``CMYK`` are converted to ``RGB``.

    Args:
        source: Path to an image file or a Pillow image.

    Returns:
        A Pillow image in ``"RGB"`` or ``"L"`` mode.

    Raises:
        FileNotFoundError: If ``source`` is a path that does not exist.
        PIL.UnidentifiedImageError: If the file is not a readable image.
    """
    image = source if isinstance(source, Image.Image) else Image.open(source)
    image = ImageOps.exif_transpose(image) or image
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def _words_from_data(data: dict[str, list[Any]]) -> list[Word]:
    """Convert ``pytesseract.image_to_data`` output into :class:`Word` objects.

    Empty tokens and non-word rows (confidence ``-1``) are skipped. Lines are
    identified by their ``(block, paragraph, line)`` triple and numbered in reading order.

    Args:
        data: The dictionary returned by ``image_to_data(..., output_type=Output.DICT)``.

    Returns:
        Recognised words in reading order.
    """
    words: list[Word] = []
    line_ids: dict[tuple[int, int, int], int] = {}
    for i, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        conf = float(data["conf"][i])
        if not text or conf < 0:
            continue
        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
        line = line_ids.setdefault(key, len(line_ids))
        words.append(
            Word(
                text=text,
                confidence=round(conf, 2),
                bbox=BoundingBox(
                    left=int(data["left"][i]),
                    top=int(data["top"][i]),
                    width=int(data["width"][i]),
                    height=int(data["height"][i]),
                ),
                line=line,
            )
        )
    return words


def extract_text(
    source: ImageSource,
    *,
    lang: str = "eng",
    preprocess_image: bool = True,
    config: PreprocessConfig | None = None,
    psm: int = 3,
    with_words: bool = True,
) -> OCRResult:
    """Extract text from an image.

    Args:
        source: Path to an image file or a Pillow image.
        lang: Tesseract language code(s), e.g. ``"eng"`` or ``"eng+ukr"``.
        preprocess_image: Run the OpenCV preprocessing pipeline before OCR.
        config: Custom preprocessing configuration (ignored if ``preprocess_image`` is False).
        psm: Tesseract page segmentation mode; ``3`` is fully automatic,
            ``6`` assumes a single uniform block of text, ``7`` a single line.
        with_words: Also collect per-word confidence and bounding boxes. Disable for
            a small speed-up when only the text is needed.

    Returns:
        An :class:`OCRResult`. Bounding boxes are in the coordinates of the image that
        was passed to Tesseract (i.e. after resizing/deskewing, if enabled).

    Raises:
        OCRError: If Tesseract is missing, the language pack is not installed, or
            Tesseract fails for another reason.

    Example:
        Find words Tesseract was unsure about::

            result = extract_text("scan.png", lang="eng+ukr")
            uncertain = [w.text for w in result.words if w.confidence < 60]
    """
    image = load_image(source)
    steps: list[str] = []
    if preprocess_image:
        prepared = preprocess(image, config)
        image, steps = prepared.image, prepared.steps

    tess_config = f"--psm {psm}"
    logger.debug("Running Tesseract (lang=%s, %s) on %dx%d image", lang, tess_config, *image.size)
    try:
        text = pytesseract.image_to_string(image, lang=lang, config=tess_config)
        words: list[Word] = []
        if with_words:
            data = pytesseract.image_to_data(
                image, lang=lang, config=tess_config, output_type=pytesseract.Output.DICT
            )
            words = _words_from_data(data)
    except pytesseract.TesseractNotFoundError as exc:
        raise OCRError("Tesseract is not installed or not on PATH") from exc
    except pytesseract.TesseractError as exc:
        raise OCRError(f"Tesseract failed: {exc.message}") from exc

    result = OCRResult(text=text.strip(), lang=lang, words=words, preprocessing=steps)
    if with_words:
        logger.info(
            "OCR finished: %d words, mean confidence %.1f",
            len(result.words),
            result.mean_confidence,
        )
    else:
        logger.info("OCR finished: %d characters", len(result.text))
    return result
