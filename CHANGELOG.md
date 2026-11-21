# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-21

### Added

- OpenCV preprocessing pipeline: grayscale, upscaling, non-local means denoising,
  minimum-area-rectangle deskew, and Otsu/adaptive thresholding, configurable via
  `PreprocessConfig`.
- `extract_text()` Tesseract wrapper returning `OCRResult` with plain text, per-word
  confidence, bounding boxes, line numbers and the list of applied preprocessing steps.
- `ocr-extract` CLI: batch processing, JSON output, language selection, threshold and
  deskew options, `--list-langs`, `--verbose` logging and output to file.
- FastAPI web demo: `POST /ocr`, `GET /languages`, `GET /health`, OpenAPI docs and a
  drag-and-drop upload page with confidence-coloured word boxes.
- Dockerfile (non-root user, health check, English and Ukrainian language packs).
- Test suite with Pillow-generated fixtures and doctests (98% coverage), strict mypy,
  ruff linting/formatting, pre-commit hooks and GitHub Actions CI on Python 3.10-3.12.

[Unreleased]: https://github.com/mzhyhan/image-ocr-extractor/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mzhyhan/image-ocr-extractor/releases/tag/v0.1.0
