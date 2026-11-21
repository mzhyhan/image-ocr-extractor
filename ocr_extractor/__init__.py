"""image-ocr-extractor: extract text from images with OpenCV preprocessing and Tesseract.

Public API:
    - :func:`extract_text` - run OCR on a path or Pillow image.
    - :class:`OCRResult`, :class:`Word`, :class:`BoundingBox` - structured results.
    - :func:`preprocess` and :class:`PreprocessConfig` - the image cleaning pipeline.
    - :class:`OCRError` - raised when OCR cannot be performed.

Example:
    >>> import ocr_extractor
    >>> ocr_extractor.__version__
    '0.1.0'
    >>> result = ocr_extractor.extract_text("receipt.jpg")  # doctest: +SKIP
"""

import logging

from ocr_extractor.ocr import BoundingBox, OCRError, OCRResult, Word, extract_text
from ocr_extractor.preprocessing import PreprocessConfig, preprocess

__all__ = [
    "BoundingBox",
    "OCRError",
    "OCRResult",
    "PreprocessConfig",
    "Word",
    "__version__",
    "extract_text",
    "preprocess",
]
__version__ = "0.1.0"

# Library best practice: let the application decide how (and whether) to log.
logging.getLogger(__name__).addHandler(logging.NullHandler())
